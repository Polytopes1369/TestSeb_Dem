#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// renderer::TransparentForwardPass's fragment stage. Phase 5 (UE5.8 parity roadmap, translucency):
// upgraded from flat sun-only Lambertian to direct sun lighting (shadowed, ComputeDirectLighting
// below) plus indirect diffuse sampled from the World Probe Grid (applied to every transparent
// material -- one 3D texture sample, cheap enough not to gate). A traced front-layer specular
// reflection (Lumen-style: single GGX-VNDF sample, HWRT/SWRT, ported from the old standalone
// TranslucentForwardPass this phase supersedes) is gated PER-MATERIAL by MaterialParams.
// hasReflections -- matching real UE5.8's "Output Reflections" material toggle (reserved for
// glass/water-like materials, not applied to every alpha-blended surface uniformly; see
// renderer::MaterialParameters::hasReflections' own comment for the exact category split).
//
// MegaLights Phase A follow-up (reconciled with the above -- `main` landed this concurrently):
// point-light shading for this pass is MegaLights' own RIS-weighted stochastic light selection + 1
// ray-traced shadow-visibility ray, exactly MegaLightsShade.comp's own algorithm, inlined directly
// here since translucent surfaces have no GBuffer entry for that compute pass to reach -- via the
// shared megalights_ris.glsl (pure weighting/reservoir math) plus this shader's own TraceShadowRay
// copy (ray-tracing code is deliberately NOT shared across shaders in this codebase, see
// megalights_ris.glsl's own header comment). This SUPERSEDES this pass's own earlier per-entity
// VSM-shadowed point-light loop -- no shadow_point_sampling.glsl here any more, MegaLights' own
// traced shadow ray covers point lights now.
//
// --- Compositing ---
// diffuseLight (sun-direct + indirect, tinted by albedo) and the MegaLights point-light term are
// both weighted by the material's own alpha via this pipeline's fixed-function "over" blend
// (srcAlpha, oneMinusSrcAlpha against the color attachment) -- NOT an explicit in-shader multiply
// (that would double-attenuate on top of the blend stage's own srcAlpha factor). When hasReflections
// is set, the reflection term is added AFTER the blend-implied weighting is accounted for and is
// NOT alpha-weighted at all (real glass's reflected light isn't attenuated by the pane's own
// transmission); the output alpha is instead boosted by the Fresnel term at grazing angles (alpha =
// max(materialAlpha, fresnelReflectance)) -- the standard real-time glass technique: a pane viewed
// edge-on should look almost fully reflective/opaque even at low base alpha.
//
// No temporal accumulation exists here (unlike ReflectionPass' ping-pong history) -- a forward-
// shaded surface has no persistent per-pixel buffer to accumulate into, so both the reflection and
// the MegaLights term are visibly noisier than their opaque-path counterparts; an accepted,
// documented simplification (matches the superseded TranslucentForwardPass's own same call).

#include "include/material_params.glsl"
#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"
#include "include/substrate_bsdf.glsl"

#define SHADOW_ATLAS_SET 0
#define SHADOW_ATLAS_BINDING 7
#define SHADOW_PAGE_TABLE_SET 0
#define SHADOW_PAGE_TABLE_BINDING 8
#define SHADOW_FEEDBACK_SET 0
#define SHADOW_FEEDBACK_BINDING 9
#define SHADOW_SUN_LEVELS_SET 0
#define SHADOW_SUN_LEVELS_BINDING 10
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

#define MEGALIGHTS_LIGHTS_SET 0
#define MEGALIGHTS_LIGHTS_BINDING 12
#include "include/megalights_ris.glsl"

// Shared by MegaLights' own TraceShadowRay AND the optional per-material reflection trace's
// TraceHWRT below -- ONE g_TLAS binding for both (renderer::SurfaceCacheRayTracingPass::
// GetTLASHandle(), see renderer::TransparentForwardPass.h's own comment).
layout(set = 0, binding = 11) uniform accelerationStructureEXT g_TLAS;

// Mirrors renderer::TransparentForwardPass.cpp's own TransparentViewParams field-for-field (std140).
// Phase 5: grew from a bare sunDirection to a camera position (feeds the reflection trace's view-
// direction reconstruction) plus sun color/intensity, matching SurfaceCacheLightingUBO's own sun
// fields (SurfaceCacheCapture.frag) so ComputeDirectLighting below is a straight port, not a
// re-derivation. No point-light fields (MegaLights supersedes them, see this file's own header
// comment) and no frameIndex (already carried by the TransparentPushConstants block below).
layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view; // Unused here (vertex-stage only) -- see TransparentForward.vert.
    mat4 proj;
    vec3 cameraPositionWorld;
    // Phase PP3: repurposes what used to be pure std140 padding -- feeds the animated procedural
    // refraction-offset noise at the end of main() below.
    float globalTime;
    vec4 sunDirectionAndIntensity; // xyz = direction (points FROM the light TOWARD the scene), w = intensity.
    vec4 sunColor;                 // rgb = color, a unused.
} g_ViewParams;

layout(std430, set = 0, binding = 6) readonly buffer MaterialParamsSSBO {
    MaterialParams params[];
} g_MaterialParams;

#define WORLD_PROBE_GRID_SET 0
#define WORLD_PROBE_GRID_BINDING 14
#define WORLD_PROBE_GRID_PARAMS_BINDING 13
#include "include/world_probe_sampling.glsl"

#define FALLBACK_GEOMETRY_SET 0
#define FALLBACK_GEOMETRY_BASE_BINDING 15
#include "include/fallback_geometry.glsl"

#include "include/mesh_sdf_trace.glsl"        // set 1
#include "include/surface_cache_sampling.glsl" // set 2

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uint inMaterialID;

layout(push_constant) uniform TransparentPushConstants {
    uint entityCount; // Dynamically uniform -- see mesh_sdf_trace.glsl's own requirement.
    uint traceMode;   // 0 = SWRT (TraceMeshSDFScene), 1 = HWRT (TraceHWRT, inline ray query).
    uint frameIndex;  // Also feeds MegaLights' own RIS candidate decorrelation (SelectLightRIS).
} pc;

layout(location = 0) out vec4 outColor;
// Phase PP3 (post-process stack roadmap): per-material procedural refraction offset -- see
// renderer::MaterialParameters::heatDistortion's own comment and this file's end-of-main() comment.
layout(location = 1) out vec2 outRefractionOffset;

// Direct lighting for this fragment: the sun only now (shadowed via SampleSunShadowVSM) -- point
// lights are MegaLights' job (see this file's own header comment) -- ported from
// SurfaceCacheCapture.frag's own identically-named function's sun term, reading g_ViewParams
// instead of that shader's SurfaceCacheLightingUBO (same sun-field layout).
//
// Substrate integration: returns just the shadowed sun RADIANCE (color * intensity * visibility),
// WITHOUT the NdotL or material multiply the pre-Substrate version applied -- EvaluateSubstrateMaterial
// (substrate_bsdf.glsl) already applies its own NdotL internally per BSDF lobe (diffuse/specular/
// fuzz), so a caller-side NdotL multiply here would double-count it. The sunNdotL<=0 early-out is
// kept purely as a cheap skip of the shadow-map sample when the sun is entirely below the surface's
// horizon (EvaluateSubstrateMaterial would itself return ~0 for every lobe in that case anyway).
vec3 ComputeSunRadiance(vec3 worldPos, vec3 n, vec3 sunDir) {
    float sunNdotL = max(dot(n, sunDir), 0.0);
    if (sunNdotL <= 0.0) {
        return vec3(0.0);
    }
    float visibility = SampleSunShadowVSM(worldPos);
    return g_ViewParams.sunColor.rgb * g_ViewParams.sunDirectionAndIntensity.w * visibility;
}

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

// Nth copy of this codebase's established inline-rayQuery HWRT trace (ReflectionTrace.comp,
// ScreenProbeTrace.comp, SurfaceCacheGIInject.comp, WorldProbeInject.comp each carry their own
// identical copy -- see ReflectionTrace.comp's own header comment on why a further copy here
// follows the existing convention rather than refactoring already-working files). Used only by the
// optional per-material reflection trace below -- MegaLights' own TraceShadowRay (right after this
// function) is a separate, cheaper any-hit-style query against the SAME g_TLAS.
bool TraceHWRT(vec3 rayOrigin, vec3 rayDir, float tMax, out uint outEntityIndex, out vec3 outLocalPos, out vec3 outLocalNormal) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, g_TLAS, gl_RayFlagsOpaqueEXT, 0xFF, rayOrigin, 1.0e-3, rayDir, tMax);
    while (rayQueryProceedEXT(rq)) {
        // Every BLAS triangle is unconditionally opaque -- nothing to confirm/reject manually.
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return false;
    }

    outEntityIndex = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
    int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    float hitT = rayQueryGetIntersectionTEXT(rq, true);
    vec3 objOrigin = rayQueryGetIntersectionObjectRayOriginEXT(rq, true);
    vec3 objDir = rayQueryGetIntersectionObjectRayDirectionEXT(rq, true);
    outLocalPos = objOrigin + objDir * hitT;

    EntityDrawRangeGpu range = g_DrawRanges[outEntityIndex];
    uint triangleFirstIndex = range.firstIndex + uint(primitiveID) * 3u;
    uint i0 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 0u];
    uint i1 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 1u];
    uint i2 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 2u];
    vec3 p0 = FetchPosition(i0);
    vec3 p1 = FetchPosition(i1);
    vec3 p2 = FetchPosition(i2);
    outLocalNormal = normalize(cross(p1 - p0, p2 - p0));
    return true;
}

// Identical to MegaLightsShade.comp's own TraceShadowRay (a boolean any-hit-style occlusion query,
// not a full hit-surface reconstruction) -- duplicated per this codebase's established per-shader
// trace-code convention, see megalights_ris.glsl's own header comment.
bool TraceShadowRay(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, g_TLAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 1.0e-3, dir, tMax);
    while (rayQueryProceedEXT(rq)) {
        // Every BLAS triangle is unconditionally opaque -- nothing to confirm/reject manually.
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

void main() {
    uint materialSlot = min(inMaterialID, MATERIAL_TABLE_SIZE - 1u);
    MaterialParams mat = g_MaterialParams.params[materialSlot];

    vec3 n = normalize(inNormal);
    // Substrate integration: the view direction EvaluateSubstrateMaterial's/
    // EvaluateSubstrateReflectionWeight's specular/Fresnel terms need -- see
    // TransparentViewParamsUBO's own cameraPositionWorld field comment above.
    vec3 viewDir = normalize(g_ViewParams.cameraPositionWorld - inWorldPos);

    // --- Sun-direct (Substrate full Slab BSDF response) + indirect diffuse (always computed --
    // cheap, applies to every transparent material regardless of hasReflections, see this file's
    // own header comment). ---
    // g_ViewParams.sunDirectionAndIntensity.xyz points FROM the light TOWARD the scene -- negate
    // for the surface-to-light direction the BSDF needs.
    vec3 sunDir = normalize(-g_ViewParams.sunDirectionAndIntensity.xyz);
    vec3 sunRadiance = ComputeSunRadiance(inWorldPos, n, sunDir);
    vec3 sunResponse = EvaluateSubstrateMaterial(mat, n, viewDir, sunDir) * sunRadiance;
    // Indirect stays a simple ambient multiply by the base slab's diffuse albedo only (no BRDF --
    // it has no single light direction), matching ClusterResolve.comp's own identical simplification
    // for its 0.15 ambient/fill term.
    vec3 indirectLighting = SampleWorldProbeGrid(inWorldPos);

    // NOT weighted by mat.alpha here -- this pipeline's fixed-function blend (srcColorBlendFactor =
    // VK_BLEND_FACTOR_SRC_ALPHA) already multiplies the WHOLE outRGB by outAlpha; an explicit
    // in-shader multiply would double-attenuate (a bug in this shader's first Phase 5 draft, fixed
    // during the MegaLights reconciliation -- see this file's own header comment).
    // kEmissiveScale -- see ClusterResolve.comp's own identical constant/comment (2026-07-17
    // recalibration): converts this codebase's small artist-authored emissive multiplier onto the
    // same real radiance scale the lit terms above now use.
    const float kEmissiveScale = 1500.0;
    vec3 outRGB = sunResponse + indirectLighting * mat.base.diffuseAlbedo + EvaluateSubstrateEmissive(mat) * kEmissiveScale;
    float outAlpha = mat.alpha;

    // --- MegaLights Phase A follow-up: RIS-selected point light + 1 shadow-visibility ray, exactly
    // MegaLightsShade.comp's own algorithm -- see this file's own header comment. Also not
    // alpha-weighted in-shader for the same reason as diffuseLight above. ---
    uint pixelSeed = uint(gl_FragCoord.y) * 65536u + uint(gl_FragCoord.x);
    uint selectedIndex;
    float invPdf;
    if (SelectLightRIS(inWorldPos, n, pixelSeed, pc.frameIndex, selectedIndex, invPdf)) {
        MegaLight light = g_Lights.lights[selectedIndex];
        vec3 toLight = light.position - inWorldPos;
        float dist = length(toLight);
        vec3 megaLightDir = toLight / max(dist, 1.0e-4);
        float megaNdotL = saturate(dot(n, megaLightDir));

        vec3 biasedOrigin = inWorldPos + n * 1.0e-2;
        bool occluded = (megaNdotL > 0.0) ? TraceShadowRay(biasedOrigin, megaLightDir, max(dist - 2.0e-2, 1.0e-3)) : true;
        float visibility = occluded ? 0.0 : 1.0;

        float distSq = max(dist * dist, 1.0e-4);
        float normalizedDist = saturate(dist / max(light.radius, 1.0e-4));
        float nd2 = normalizedDist * normalizedDist;
        float windowSq = 1.0 - nd2 * nd2;
        float window = saturate(windowSq * windowSq);

        // Substrate: EvaluateSubstrateMaterial already applies its own NdotL internally (see
        // ComputeSunRadiance's own comment above) -- megaNdotL is kept only for the occlusion
        // early-out check above, no longer multiplied into the final radiance here. light.intensity
        // is real luminous intensity in CANDELA (renderer::MegaLight's own comment, 2026-07-17
        // recalibration) -- `intensity / distSq` is the standard inverse-square illuminance-at-a-
        // point formula; EvaluateSubstrateMaterial's own contract already bakes in the /PI
        // Lambertian normalization, so no extra factor is needed here.
        outRGB += EvaluateSubstrateMaterial(mat, n, viewDir, megaLightDir) * light.color * light.intensity / distSq * window * visibility * invPdf;
    }

    // --- Optional front-layer specular reflection (Lumen-style, single GGX-VNDF sample) -- gated
    // per-material, see this file's own header comment. ---
    if (mat.hasReflections > 0.5) {
        // Substrate integration: Fresnel-only "Lumen Performance mode" reflection weight,
        // generalized to Substrate's vertical layering (base+top slabs) -- see
        // substrate_bsdf.glsl's own EvaluateSubstrateReflectionWeight comment. viewDir already
        // computed above (shared with the sun/MegaLights BSDF terms).
        vec3 fresnel = EvaluateSubstrateReflectionWeight(mat, n, viewDir);

        vec3 tangent, bitangent;
        BuildTangentBasis(n, tangent, bitangent);
        vec3 viewDirTangentSpace = vec3(dot(viewDir, tangent), dot(viewDir, bitangent), dot(viewDir, n));

        // Per-pixel-per-frame decorrelated jitter (gl_FragCoord takes the place of
        // ReflectionTrace.comp's gl_GlobalInvocationID -- same hashFloat construction).
        uint64_t pixelSeed64 = uint64_t(uint(gl_FragCoord.x)) | (uint64_t(uint(gl_FragCoord.y)) << 32);
        vec2 xi = vec2(
            hashFloat(pixelSeed64, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0x517CC1B7UL),
            hashFloat(pixelSeed64, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0xBF58476D1CE4E5B9UL));

        vec3 halfVectorTangentSpace = SampleGGXVNDF(xi, viewDirTangentSpace, mat.base.roughness);
        vec3 halfVectorWorld = normalize(
            tangent * halfVectorTangentSpace.x + bitangent * halfVectorTangentSpace.y + n * halfVectorTangentSpace.z);
        vec3 rayDir = reflect(-viewDir, halfVectorWorld);

        vec3 reflectionRadiance = vec3(0.0);
        // Grazing-angle GGX-VNDF edge case (sampled half-vector below the macro-surface plane) is
        // left as a miss (reflectionRadiance stays 0.0), same handling as ReflectionTrace.comp's own.
        if (dot(rayDir, n) > 0.0) {
            vec3 biasedOrigin = inWorldPos + n * 1.0e-2;

            uint hitEntityIndex;
            vec3 hitLocalPos, hitLocalNormal;
            bool hit;
            if (pc.traceMode == 0u) {
                hit = TraceMeshSDFScene(biasedOrigin, rayDir, kFarDistance, pc.entityCount, hitEntityIndex, hitLocalPos, hitLocalNormal);
            } else {
                hit = TraceHWRT(biasedOrigin, rayDir, kFarDistance, hitEntityIndex, hitLocalPos, hitLocalNormal);
            }

            if (hit) {
                EntityInfo hitEntity = g_Entities[hitEntityIndex];
                reflectionRadiance = SampleCardRadiance(hitLocalPos, hitLocalNormal, hitEntity.firstCardIndex, hitEntity.cardCount);
            }
        }

        // Reflected light is NOT alpha-weighted (real glass's reflection isn't attenuated by the
        // pane's own transmission) and boosts the output alpha at grazing angles -- see this
        // file's own header comment.
        outRGB += fresnel * reflectionRadiance;
        float fresnelScalar = clamp(max(fresnel.r, max(fresnel.g, fresnel.b)), 0.0, 1.0);
        outAlpha = clamp(max(mat.alpha, fresnelScalar), 0.0, 1.0);
    }

    outColor = vec4(outRGB, outAlpha);

    // --- Phase PP3: procedural, animated refraction offset (Heat Distortion & Refraction) --
    // UE5.8-parity material-authored mechanism: this fragment writes a per-pixel screen-space UV
    // offset into g_RefractionOffset (this pass' own second color attachment), which renderer::
    // PostProcessPass's composite shader later samples and uses to distort the UV it reads the HDR
    // scene color through -- exactly like a real UE5.8 translucent material's "Refraction" input,
    // not a single global post-process knob. Two independently-scrolling value-noise fields (world-
    // space XZ, offset by g_ViewParams.globalTime) drive the X/Y offset components -- cheap, no
    // texture asset, reads as a believable rising-heat shimmer for any material with
    // MaterialParams.heatDistortion > 0 (see MaterialParameterTable.h's own category comment for
    // which materials roll one).
    if (mat.heatDistortion > 0.0) {
        vec2 flowUV = inWorldPos.xz * 0.35 + vec2(0.0, g_ViewParams.globalTime * 0.6);
        float noiseX = Hash(flowUV) - 0.5;
        float noiseY = Hash(flowUV * 1.7 + vec2(19.3, 7.1) + g_ViewParams.globalTime * 0.4) - 0.5;
        outRefractionOffset = vec2(noiseX, noiseY) * mat.heatDistortion * 0.015;
    } else {
        outRefractionOffset = vec2(0.0);
    }
}
