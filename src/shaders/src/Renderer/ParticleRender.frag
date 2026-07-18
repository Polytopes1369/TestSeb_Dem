#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// Particle system Subtask 4+5 (see ParticleRender.vert's own header comment for the full billboard
// contract this fragment shader receives). Effects on top of the plain per-particle color:
//
// --- Procedural sprite shape ---
// This project has zero on-disk texture assets (CLAUDE.md's "no data in the .exe" constraint --
// every material/noise/sky/cloud in this codebase is generated analytically, see e.g.
// procedural_material.glsl / atmos_clouds_density.glsl), so the plan doc's literal "sample a
// texture-atlas" instruction is adapted into a plain analytic soft-circle mask in UV space instead
// of a loaded sprite sheet -- consistent with every other visual system in this renderer.
//
// --- Soft particles ---
// Reconstructs the opaque scene's world position under this fragment from renderer::
// ClusterResolvePass's own GBuffer depth copy (bound once at Init(), see renderer::
// ParticleSystemPass::Init's own comment) and fades the sprite out as it nears an intersection with
// that surface, instead of hard-clipping at the fixed-function depth test's binary pass/fail edge.
// This codebase has no existing linear-depth-reconstruction helper (every other depth consumer works
// in raw NDC space directly, see e.g. SDFRayMarch.comp's own SampleClipmap) -- this shader
// reconstructs a genuine WORLD position via the inverse view-projection matrix (mirroring
// SSRFallback.comp's own ReconstructWorldPos) and fades on the resulting camera-space DISTANCE
// difference, which stays physically meaningful (a fixed world-unit fade band) at every depth,
// unlike a raw (and highly nonlinear, reversed-Z) NDC delta would.
//
// --- Subtask 5: sun (VSM-shadowed) + indirect diffuse (World Probe Grid) ---
// A billboard has no real surface normal, so this is deliberately NOT a BRDF evaluation (no
// NdotL/Henyey-Greenstein phase term -- the plan doc's own "simplified Lambertian" allowance) --
// just the shadowed sun radiance plus indirect diffuse irradiance, both isotropic, added together
// and modulating the particle's own albedo. Same VSM include block as TransparentForward.frag /
// SurfaceCacheCapture.frag (shadow_page_table/feedback/atlas_sampling/sun_sampling.glsl); same
// world_probe_sampling.glsl macro contract every other consumer (TransparentForward.frag,
// Tessellation.frag) already uses.
//
// --- Subtask 5: heat-shimmer refraction ---
// When g_Params.heatShimmerStrength > 0 (a per-draw-call toggle -- see renderer::
// ParticleSystemPass::RecordDraw's own comment for why this stays per-draw-call, not per-particle,
// even after the multi-emitter roadmap's GpuParticle.emitterIndex addition: a demoscene emitter is
// realistically one thermal "kind" or another, never a per-particle mix), writes a
// small time-varying wobble into this pass' second color output -- renderer::
// TransparentForwardPass's OWN shared g_RefractionOffset image (this codebase's first SECOND writer
// of that image, see ParticleSystemPass::RecordDraw's own comment on why the load/store discipline
// there matters), later read by PostProcessComposite.comp exactly like glass's own contribution.
//
// --- Niagara-parity render-integration roadmap (D1/D2/D3/D5) ---
// Four additions layered on top of Subtask 5's existing sun+World-Probe-Grid combination, all still
// isotropic/no-BRDF (a billboard has no real surface normal, see this file's own header comment
// above -- unchanged rationale for every new light term below):
//   D1 -- MegaLights: one RIS-selected stochastic point light + its own mandatory shadow-visibility
//     ray (TraceShadowRay below), exactly mirroring TransparentForward.frag's own inlined MegaLights
//     shading (same megalights_ris.glsl contract, same "no spatial BVH pool -- fall back to the
//     full-population draw" choice that shader already makes for its own non-GBuffer forward path,
//     see this file's own SelectLightRIS call site for why that choice is mirrored here too).
//   D2 -- direct Surface Cache sampling was investigated and DELIBERATELY SCOPED DOWN: surface_cache_
//     sampling.glsl/mesh_sdf_trace.glsl hardcode descriptor set=2/set=1 respectively (no caller-
//     defined-macro override, unlike every other shared binding header this file already includes),
//     which collide with this pipeline's own pre-existing set 2 (ParticleRenderParamsUBO/g_SceneDepth)
//     and set 1 (SortedPairsBuffer, ParticleRender.vert's own binding) -- resolving that would mean
//     either renumbering this pipeline's ENTIRE existing 4-set layout (high regression risk across
//     vert+frag+CPU, touching the sort-buffer binding contract this roadmap was explicitly told not
//     to risk) or converting those 2 shared headers to caller-defined set macros (touching all 9 of
//     their existing includers across the core GI pipeline -- SurfaceCacheGIInject/WorldProbeInject/
//     ReflectionTrace/ScreenProbeTrace/SurfaceCacheHWRT among them -- well outside this workstream's
//     scope and risk budget). Implemented instead, at ZERO new descriptor bindings: when the
//     EXISTING soft-particle depth reconstruction below already found nearby opaque geometry, a
//     SECOND World Probe Grid tap is taken at that reconstructed surface point and blended with the
//     particle's own tap, weighted toward the surface the closer the particle is to intersecting it
//     -- a real, if modest, "sharper toward geometry" bias using resources already bound, not a
//     genuine per-card radiance fetch.
//   D3 -- point-light VSM shadows: shadow_point_sampling.glsl's existing SamplePointShadowVSM,
//     looped over g_PointLights (this pass' own small per-frame UBO, mirroring renderer::SceneLights
//     ::pointLights exactly) -- the sun was the only shadowed light this shader knew about before.
//   D5 -- volumetric fog: samples renderer::AtmosVolumetricFogPass's own integrated froxel grid at
//     this fragment's (screen UV, view-space depth) -- the SAME (uv, viewZ) contract
//     PostProcessComposite.comp's own ApplyVolumetricFog reader already established -- and tints/
//     modulates the particle's final lit color by it, so a particle embedded in dense fog reads as
//     fog-tinted instead of fog being invisible to (drawn "through") the particle layer.

// Only needed here for the kParticleKind* constants this file's own shapeMask branch below reads --
// the actual ParticleBuffer/DeadList/AliveList/Counter SSBO bindings this header also declares go
// unused in this stage (ParticleRender.vert is set 0's only real reader), which is harmless: set 0's
// own VkDescriptorSetLayoutBinding stage flags already include VK_SHADER_STAGE_FRAGMENT_BIT (see
// renderer::ParticleSystemPass::Init's own STEP 3 comment), so redeclaring the same bindings here is
// legal and costs nothing at runtime.
#include "include/ParticleCommon.glsl"

layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
    vec3 sunDirection; float sunIntensity; // sunDirection points FROM the light TOWARD the scene.
    vec3 sunColor; float _pad3;
    // D5: repurposes what used to be 3 unused trailing pad floats -- flat scalars (not vec3), same
    // "avoid std140's vec3-alignment padding diverging from the CPU struct's own flat layout"
    // convention as every other camera-basis field in this exact UBO.
    float heatShimmerStrength; float cameraForwardX, cameraForwardY, cameraForwardZ;
    // D1: new trailing 16-byte block -- decorrelates MegaLights' own RIS candidate draw per frame.
    uint frameIndex; float _pad7, _pad8, _pad9;
} g_Params;

// renderer::ClusterResolvePass::GetOutputDepthView() -- the SAMPLED GBuffer depth copy (R32_SFLOAT,
// raw NDC z, reversed-Z: near = 1.0, far = 0.0), NOT the real depth-stencil attachment this pipeline
// depth-tests against (that one is bound as this draw's actual VkRenderingAttachmentInfo depth
// attachment, read-only, see RecordDraw's own comment -- a depth ATTACHMENT cannot simultaneously be
// sampled by the same draw's own fragment shader, hence needing this separate sampled copy).
layout(set = 2, binding = 1) uniform sampler2D g_SceneDepth;

#define SHADOW_PAGE_TABLE_SET 3
#define SHADOW_PAGE_TABLE_BINDING 0
#define SHADOW_FEEDBACK_SET 3
#define SHADOW_FEEDBACK_BINDING 1
#define SHADOW_ATLAS_SET 3
#define SHADOW_ATLAS_BINDING 2
#define SHADOW_SUN_LEVELS_SET 3
#define SHADOW_SUN_LEVELS_BINDING 3
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

#define WORLD_PROBE_GRID_SET 3
#define WORLD_PROBE_GRID_BINDING 4
#define WORLD_PROBE_GRID_PARAMS_BINDING 5
#include "include/world_probe_sampling.glsl"

// D1 (MegaLights sampling for particles): same set-3 "lighting" descriptor set every VSM/World-
// Probe-Grid binding above already lives in -- see renderer::ParticleSystemPass::Init's own STEP 6b
// comment for the full binding-index map. math_utils.glsl is required by megalights_ris.glsl (saturate()/
// Halton23()) -- not previously included by this shader.
#include "include/math_utils.glsl"
#define MEGALIGHTS_LIGHTS_SET 3
#define MEGALIGHTS_LIGHTS_BINDING 6
#include "include/megalights_ris.glsl"

// D1: shared g_TLAS -- MegaLights' own mandatory shadow-visibility ray (TraceShadowRay below) is
// this shader's only ray-tracing consumer (unlike TransparentForward.frag, this pass has no
// optional per-material reflection trace, so no second consumer of this same binding exists here).
layout(set = 3, binding = 7) uniform accelerationStructureEXT g_TLAS;

// D3 (point-light VSM shadows): renderer::VirtualShadowMapPass's own per-light 6-face cube VSMs --
// reuses the SAME SHADOW_SUN_LEVEL_COUNT/SHADOW_PAGES_PER_AXIS/page-table/feedback/atlas-sampling
// machinery the sun's own shadow_sun_sampling.glsl include above already pulled in.
#define SHADOW_POINT_FACES_SET 3
#define SHADOW_POINT_FACES_BINDING 8
#include "include/shadow_point_sampling.glsl"

// D3: this pass' own small per-frame point-light UBO -- mirrors renderer::SceneLights::pointLights
// (LightingTypes.h) field-for-field, flat scalars (not vec3/PointLight-array-of-struct) for the same
// std140-padding-avoidance reason as g_Params' own camera-basis fields above. Uploaded fresh every
// RecordDraw() call (ordinary per-frame CPU light data, unlike the borrowed GPU-owned resources this
// set otherwise only binds once -- see renderer::ParticleSystemPass::Init's own STEP 6b comment).
#define PARTICLE_MAX_POINT_LIGHTS 8u // Must match renderer::kMaxPointLights (LightingTypes.h) exactly.
struct ParticlePointLight {
    vec3 position; float _padA;
    vec3 color; float intensity;
    float radius; float _padB, _padC, _padD;
};
layout(std140, set = 3, binding = 9) uniform ParticlePointLightsUBO {
    ParticlePointLight lights[PARTICLE_MAX_POINT_LIGHTS];
    uint pointLightCount;
} g_PointLights;

// D5 (volumetric fog interaction): renderer::AtmosVolumetricFogPass's own integrated froxel grid --
// same (uv, viewZ) reader contract PostProcessComposite.comp's own ApplyVolumetricFog already uses.
layout(set = 3, binding = 10) uniform sampler3D g_VolumetricFog;
#include "include/atmos_volumetric_fog_mapping.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inNormalizedAge;
layout(location = 4) flat in uint inKind; // kParticleKindEmber/Rain/Snow (ParticleCommon.glsl) -- see this file's own shapeMask comment below.

// D1: identical to MegaLightsShade.comp's/TransparentForward.frag's own TraceShadowRay (a boolean
// any-hit-style occlusion query, not a full hit-surface reconstruction) -- duplicated per this
// codebase's established per-shader trace-code convention, see megalights_ris.glsl's own header
// comment.
bool TraceShadowRay(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, g_TLAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 1.0e-3, dir, tMax);
    while (rayQueryProceedEXT(rq)) {
        // Every BLAS triangle is unconditionally opaque -- nothing to confirm/reject manually.
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

layout(location = 0) out vec4 outColor;
// renderer::TransparentForwardPass's own shared heat-distortion target -- see this file's own
// header comment. Written every fragment (even when heatShimmerStrength == 0, in which case it is
// simply (0,0) -- a plain overwrite, blendEnable=FALSE for this attachment, so an explicit zero is
// required rather than "just don't write," which would leave whatever a PREVIOUS forward pass wrote
// at this exact pixel untouched -- wrong, since a non-shimmering particle covering that pixel should
// suppress any earlier distortion there, exactly like an opaque particle would).
layout(location = 1) out vec2 outRefractionOffset;

void main() {
    vec2 centered = inUV * 2.0 - 1.0; // Both axes in [-1, 1]; y runs along ParticleRender.vert's rotatedOffset "length" axis, x across its "width" axis.
    float shapeMask;
    if (inKind == kParticleKindRain) {
        // Precipitation feature: rain is a thin streak, not a round sprite -- fade out fast across the
        // (already-thin, see SpawnPrecipitationParticle's own size.x) width, and only softly at the
        // very tip of the (long, size.y) length, so it reads as a falling line rather than a blob.
        shapeMask = smoothstep(1.0, 0.15, abs(centered.x)) * smoothstep(1.0, 0.9, abs(centered.y));
    } else {
        // Embers and snow: the original plain analytic soft-circle mask (this project has zero on-disk
        // texture assets, see this file's own header comment) -- snow reuses it unmodified, a soft
        // round flake needs no kind-specific change here.
        shapeMask = smoothstep(1.0, 0.0, length(centered));
    }
    if (shapeMask <= 0.0) {
        discard; // Outside the sprite's soft shape -- skip the (otherwise wasted) work below.
    }

    vec2 screenUV = gl_FragCoord.xy / g_Params.viewportSize;
    float sceneNdcDepth = texture(g_SceneDepth, screenUV).r;

    // sceneNdcDepth <= 0.0 means sky/no-geometry-hit at this pixel (see renderer::ClusterResolvePass's
    // own g_OutputDepth clear-to-0 convention) -- nothing to soft-fade against, draw at full strength.
    float softFade = 1.0;
    // D2 (Surface Cache sampling, scoped down -- see this file's own header comment): the
    // reconstructed nearby-opaque-surface world position, valid only when hasNearbySurface is true.
    // Reused below to sharpen the World Probe Grid sample toward whatever surface this particle is
    // closest to intersecting, at zero extra descriptor bindings (this is the SAME reconstruction
    // the soft-particle fade above already needs).
    bool hasNearbySurface = false;
    vec3 sceneWorldPos = vec3(0.0);
    float sceneDist = 0.0;
    float particleDist = length(inWorldPos - g_Params.cameraPosition);
    if (sceneNdcDepth > 0.0) {
        vec4 clip = vec4(screenUV * 2.0 - 1.0, sceneNdcDepth, 1.0);
        vec4 sceneWorld4 = g_Params.invViewProj * clip;
        sceneWorldPos = sceneWorld4.xyz / sceneWorld4.w;
        hasNearbySurface = true;

        sceneDist = length(sceneWorldPos - g_Params.cameraPosition);
        softFade = clamp((sceneDist - particleDist) / max(g_Params.softFadeDistance, 1.0e-4), 0.0, 1.0);
    }

    // Fade out over the last 20% of remaining life rather than vanishing the instant a particle is
    // recycled server-side (see ParticleRender.vert's own outNormalizedAge comment).
    float lifeFade = smoothstep(0.0, 0.2, inNormalizedAge);

    float alpha = inColor.a * shapeMask * softFade * lifeFade;

    // --- Sun (VSM-shadowed) + indirect diffuse (World Probe Grid) -- see this file's own header
    // comment for why there is no NdotL/phase-function term. A small constant floor keeps a
    // particle sitting in full shadow with no probe coverage nearby from going pure black. ---
    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    vec3 sunRadiance = g_Params.sunColor * g_Params.sunIntensity * sunVisibility;
    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos);

    // D2 (scoped-down Surface Cache supplement -- see this file's own header comment): blend in a
    // SECOND World Probe Grid tap taken at the reconstructed nearby-surface position, weighted by
    // how close this particle actually is to intersecting it (the same (sceneDist - particleDist)
    // term softFade above already computes) -- a particle floating far from any surface keeps its
    // own single tap unchanged (blendToSurface -> 0), one about to visually merge into geometry
    // leans toward that geometry's own local probe sample instead (blendToSurface -> 1).
    if (hasNearbySurface) {
        float surfaceProximity = 1.0 - clamp((sceneDist - particleDist) / max(g_Params.softFadeDistance, 1.0e-4), 0.0, 1.0);
        float blendToSurface = surfaceProximity * surfaceProximity; // Smoother falloff than linear.
        vec3 surfaceIndirect = SampleWorldProbeGrid(sceneWorldPos);
        indirectDiffuse = mix(indirectDiffuse, surfaceIndirect, blendToSurface);
    }

    vec3 lighting = sunRadiance + indirectDiffuse + vec3(0.02);

    // --- D1: MegaLights -- one RIS-selected stochastic point light + its own mandatory shadow-
    // visibility ray, exactly MegaLightsShade.comp's/TransparentForward.frag's own algorithm (see
    // this file's own header comment). No spatial BVH pool here either (mirrors TransparentForward
    // .frag's own choice for its non-GBuffer forward path) -- SelectLightRIS falls back to its
    // original full-population draw unchanged. ---
    {
        uint pixelSeed = uint(gl_FragCoord.y) * 65536u + uint(gl_FragCoord.x);
        uint selectedIndex;
        float invPdf;
        uint noSpatialPool[kMegaLightsSpatialPoolCapacity];
        if (SelectLightRIS(inWorldPos, vec3(0.0, 1.0, 0.0), pixelSeed, g_Params.frameIndex, noSpatialPool, 0u, selectedIndex, invPdf)) {
            MegaLight light = g_Lights.lights[selectedIndex];
            vec3 toLight = light.position - inWorldPos;
            float dist = length(toLight);
            vec3 megaLightDir = toLight / max(dist, 1.0e-4);

            bool occluded = TraceShadowRay(inWorldPos, megaLightDir, max(dist - 2.0e-2, 1.0e-3));
            float visibility = occluded ? 0.0 : 1.0;

            float distSq = max(dist * dist, 1.0e-4);
            float normalizedDist = clamp(dist / max(light.radius, 1.0e-4), 0.0, 1.0);
            float nd2 = normalizedDist * normalizedDist;
            float windowSq = 1.0 - nd2 * nd2;
            float window = clamp(windowSq * windowSq, 0.0, 1.0);

            lighting += light.color * (light.intensity / distSq) * window * visibility * invPdf;
        }
    }

    // --- D3: point-light VSM shadows -- same isotropic, no-BRDF convention as the sun/MegaLights
    // terms above; MegaLights already covers stochastic MANY-light sampling, this instead gives the
    // handful of authored point lights (renderer::SceneLights::pointLights) a REAL per-light shadow
    // via their own dedicated VSM cube, exactly like opaque geometry already gets. ---
    for (uint i = 0u; i < g_PointLights.pointLightCount; ++i) {
        ParticlePointLight pl = g_PointLights.lights[i];
        vec3 toLight = pl.position - inWorldPos;
        float dist = length(toLight);
        float distSq = max(dist * dist, 1.0e-4);

        float normalizedDist = clamp(dist / max(pl.radius, 1.0e-4), 0.0, 1.0);
        float nd2 = normalizedDist * normalizedDist;
        float windowSq = 1.0 - nd2 * nd2;
        float window = clamp(windowSq * windowSq, 0.0, 1.0);
        if (window <= 0.0) {
            continue; // Outside this light's own falloff radius -- skip the (otherwise wasted) shadow sample.
        }

        float visibility = SamplePointShadowVSM(i, pl.position, inWorldPos);
        lighting += pl.color * (pl.intensity / distSq) * window * visibility;
    }

    outColor = vec4(inColor.rgb * lighting, alpha);

    // --- D5: volumetric fog -- tints/modulates this particle's own already-lit color by
    // renderer::AtmosVolumetricFogPass's integrated froxel sample at THIS PARTICLE's OWN (screen
    // UV, view-space depth), mirroring PostProcessComposite.comp's own ApplyVolumetricFog exactly
    // (fogSample.a = transmittance, fogSample.rgb = in-scattered light) -- NOT weighted by alpha
    // in-shader (this pipeline's fixed-function blend already multiplies the whole outRGB by
    // outAlpha at composite time, same "avoid double-attenuating" reasoning TransparentForward.
    // frag's own header comment documents for its own light terms).
    //
    // KNOWN, ACCEPTED LIMITATION: renderer::PostProcessPass's own global ApplyVolumetricFog call
    // (PostProcessComposite.comp) still runs once more, later, over the WHOLE composited frame
    // (particles included, since they draw before post-process) -- but it reconstructs viewZ from
    // the OPAQUE GBuffer depth only (particles never write depth, see this pass' own
    // depthWriteEnable=FALSE pipeline state), which is the WRONG (too-far) depth for any pixel a
    // particle nearer than the background covers. This per-particle pass fixes that specific case
    // using the particle's OWN correct depth; the global pass then re-applies its own (background-
    // depth) fog on top, a mild double-application in exchange for a strictly more correct near-
    // camera term -- fixing the global pass to be particle-depth-aware would require touching
    // renderer::PostProcessPass itself, out of scope for this workstream (see this file's own D2
    // scope-down comment for the same "avoid touching shared passes outside this pipeline" reasoning).
    {
        float viewZ = dot(inWorldPos - g_Params.cameraPosition, vec3(g_Params.cameraForwardX, g_Params.cameraForwardY, g_Params.cameraForwardZ));
        float froxelW = ViewZToFroxelW(viewZ);
        vec4 fogSample = texture(g_VolumetricFog, vec3(screenUV, froxelW));
        outColor.rgb = outColor.rgb * fogSample.a + fogSample.rgb;
    }

    if (g_Params.heatShimmerStrength > 0.0) {
        vec2 shimmer = vec2(
            sin(g_Params.globalTime * 3.0 + inWorldPos.x * 5.0 + inWorldPos.z * 3.0),
            cos(g_Params.globalTime * 2.7 + inWorldPos.z * 5.0 - inWorldPos.x * 3.0));
        outRefractionOffset = shimmer * g_Params.heatShimmerStrength * shapeMask * alpha;
    } else {
        outRefractionOffset = vec2(0.0);
    }
}
