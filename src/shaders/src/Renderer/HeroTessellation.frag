#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// Phase 7a (UE5.8 parity roadmap, hero asset tessellation): same "never appears in
// ClusterResolvePass's own GBuffer" constraint TransparentForward.frag's own header comment
// documents (this material's entity is unconditionally excluded from the opaque Nanite path via
// core::EntityFlags::IsTransparent, see VulkanContext::BuildEntityData()'s own comment) -- so this
// fragment shader does its OWN inline direct+MegaLights+indirect+optional-reflection composition,
// reusing the exact same MegaLights RIS integration TransparentForward.frag already established
// (see that shader's own header comment for the full rationale: translucent/hero surfaces have no
// GBuffer entry for MegaLightsPass' own compute-pass composite to reach, so its algorithm is
// inlined here via the shared megalights_ris.glsl plus this shader's own TraceShadowRay copy).
// OPAQUE, unlike TransparentForward.frag: no Fresnel-boosted alpha, outColor.a is always 1.0.
// `inWorldNormal` is already the DISPLACED surface normal (HeroTessellation.tese's own
// ComputeDisplacedNormal), not a flat undisplaced one.

#include "include/material_params.glsl"
#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"

#define SHADOW_ATLAS_SET 0
#define SHADOW_ATLAS_BINDING 1
#define SHADOW_PAGE_TABLE_SET 0
#define SHADOW_PAGE_TABLE_BINDING 2
#define SHADOW_FEEDBACK_SET 0
#define SHADOW_FEEDBACK_BINDING 3
#define SHADOW_SUN_LEVELS_SET 0
#define SHADOW_SUN_LEVELS_BINDING 4
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

#define MEGALIGHTS_LIGHTS_SET 0
#define MEGALIGHTS_LIGHTS_BINDING 10
#include "include/megalights_ris.glsl"

// Shared by MegaLights' own TraceShadowRay AND the optional reflection trace's TraceHWRT below --
// ONE g_TLAS binding for both, same convention TransparentForward.frag already established.
layout(set = 0, binding = 9) uniform accelerationStructureEXT g_TLAS;

// This pass' own small sun-lighting UBO -- see renderer::HeroTessellationPass.cpp's own
// HeroLightingUBO comment for why viewProj/cameraPos are NOT duplicated here (already in the push
// constants below, needed by the vertex/tessellation stages too).
layout(std140, set = 0, binding = 0) uniform HeroLightingUBO {
    vec4 sunDirectionAndIntensity; // xyz = direction (points FROM the light TOWARD the scene), w = intensity.
    vec4 sunColor;                 // rgb = color, a unused.
} g_Lighting;

// Single-element buffer -- this pass only ever shades ONE fixed material (see
// renderer::HeroTessellationPass.cpp's own `heroMaterial` parameter comment), unlike
// TransparentForward.frag's own per-fragment materialID-indexed array.
layout(std430, set = 0, binding = 7) readonly buffer MaterialParamsSSBO {
    MaterialParams mat;
} g_MaterialParams;

#define WORLD_PROBE_GRID_SET 0
#define WORLD_PROBE_GRID_BINDING 5
#define WORLD_PROBE_GRID_PARAMS_BINDING 6
#include "include/world_probe_sampling.glsl"

// binding 8 == g_EntityTransforms, declared by HeroTessellation.vert only -- this fragment shader
// never needs it, the vertex/tessellation stages already resolved the (displaced) world position.

// HWRT resources (own set 0 bindings 11-13, NOT part of SurfaceCacheTraceContext's set 1/2 --
// mirrors TransparentForward.frag's own equivalent bindings, itself mirroring
// ReflectionTrace.comp's own pattern).
struct FallbackVertexGpu { float posX, posY, posZ; float normX, normY, normZ; float u, v; };
layout(std430, set = 0, binding = 11) readonly buffer FallbackVertexBuffer { FallbackVertexGpu g_Vertices[]; };
layout(std430, set = 0, binding = 12) readonly buffer FallbackIndexBuffer { uint g_Indices[]; };
struct EntityDrawRangeGpu { int vertexOffset; uint firstIndex; uint indexCount; uint _pad; };
layout(std430, set = 0, binding = 13) readonly buffer EntityDrawRangeBuffer { EntityDrawRangeGpu g_DrawRanges[]; };

#include "include/mesh_sdf_trace.glsl"        // set 1
#include "include/surface_cache_sampling.glsl" // set 2

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;

// Byte-for-byte mirror of HeroTessellationConstants in .vert/.tesc/.tese -- same full struct
// declared identically in every stage sharing this one push-constant range.
layout(push_constant) uniform HeroTessellationConstants {
    mat4 viewProj;
    float cameraPositionWorldX, cameraPositionWorldY, cameraPositionWorldZ;
    float _pad0;
    uint entityID;
    uint traceMode;
    uint frameIndex;
    uint entityCount;
    float viewportWidth, viewportHeight;
    float displacementScale;
    float _pad1;
} pc;

layout(location = 0) out vec4 outColor;

// Sun-only direct lighting (shadowed via SampleSunShadowVSM) -- point lights are MegaLights' job
// (see this file's own header comment) -- identical formula to TransparentForward.frag's own
// ComputeDirectLighting, reading g_Lighting instead of that shader's TransparentViewParamsUBO.
vec3 ComputeDirectLighting(vec3 worldPos, vec3 n) {
    vec3 sunDir = normalize(-g_Lighting.sunDirectionAndIntensity.xyz);
    float sunNdotL = max(dot(n, sunDir), 0.0);
    if (sunNdotL <= 0.0) {
        return vec3(0.0);
    }
    float visibility = SampleSunShadowVSM(worldPos);
    return g_Lighting.sunColor.rgb * g_Lighting.sunDirectionAndIntensity.w * sunNdotL * visibility;
}

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

// Nth copy of this codebase's established inline-rayQuery HWRT trace (ReflectionTrace.comp,
// ScreenProbeTrace.comp, SurfaceCacheGIInject.comp, WorldProbeInject.comp, TransparentForward.frag
// each carry their own identical copy -- see TransparentForward.frag's own header comment on why a
// further copy here follows the existing convention rather than refactoring already-working
// files). Used only by the optional per-material reflection trace below -- MegaLights' own
// TraceShadowRay (right after this function) is a separate, cheaper any-hit-style query against
// the SAME g_TLAS.
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
    vec3 n = normalize(inWorldNormal);
    MaterialParams mat = g_MaterialParams.mat;

    // --- Sun-direct + indirect diffuse (always computed -- cheap, one 3D texture sample). ---
    vec3 directLighting = ComputeDirectLighting(inWorldPos, n);
    vec3 indirectLighting = SampleWorldProbeGrid(inWorldPos);
    vec3 diffuseAlbedo = mat.baseColor * (1.0 - mat.metallic);
    vec3 diffuseLight = diffuseAlbedo * (directLighting + indirectLighting);

    // Fully opaque -- no blend stage involved (blendEnable = false, see
    // renderer::HeroTessellationPass's own class comment), so no alpha-weighting concern here.
    vec3 outRGB = diffuseLight + mat.emissive;

    // --- MegaLights: RIS-selected point light + 1 shadow-visibility ray, exactly
    // MegaLightsShade.comp's own algorithm -- see this file's own header comment. ---
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

        outRGB += diffuseAlbedo * light.color * light.intensity * megaNdotL / distSq * window * visibility * invPdf;
    }

    // --- Optional front-layer specular reflection (Lumen-style, single GGX-VNDF sample) -- gated
    // per-material via mat.hasReflections (kHeroMaterialID's own recipe has it on, showcasing the
    // same technique TransparentForwardPass's "Transparent: clear/glass-like" category uses). ---
    if (mat.hasReflections > 0.5) {
        vec3 cameraPositionWorld = vec3(pc.cameraPositionWorldX, pc.cameraPositionWorldY, pc.cameraPositionWorldZ);
        vec3 viewDir = normalize(cameraPositionWorld - inWorldPos);
        float NdotV = max(dot(n, viewDir), 0.0);
        vec3 F0 = mix(vec3(0.04), mat.baseColor, mat.metallic);
        vec3 fresnel = F_Schlick(F0, NdotV);

        vec3 tangent, bitangent;
        BuildTangentBasis(n, tangent, bitangent);
        vec3 viewDirTangentSpace = vec3(dot(viewDir, tangent), dot(viewDir, bitangent), dot(viewDir, n));

        // Per-pixel-per-frame decorrelated jitter (gl_FragCoord takes the place of
        // ReflectionTrace.comp's gl_GlobalInvocationID -- same hashFloat construction).
        uint64_t pixelSeed64 = uint64_t(uint(gl_FragCoord.x)) | (uint64_t(uint(gl_FragCoord.y)) << 32);
        vec2 xi = vec2(
            hashFloat(pixelSeed64, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0x517CC1B7UL),
            hashFloat(pixelSeed64, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0xBF58476D1CE4E5B9UL));

        vec3 halfVectorTangentSpace = SampleGGXVNDF(xi, viewDirTangentSpace, mat.roughness);
        vec3 halfVectorWorld = normalize(
            tangent * halfVectorTangentSpace.x + bitangent * halfVectorTangentSpace.y + n * halfVectorTangentSpace.z);
        vec3 rayDir = reflect(-viewDir, halfVectorWorld);

        // Grazing-angle GGX-VNDF edge case (sampled half-vector below the macro-surface plane) is
        // left as a miss, same handling as TransparentForward.frag's own.
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
                vec3 reflectionRadiance = SampleCardRadiance(hitLocalPos, hitLocalNormal, hitEntity.firstCardIndex, hitEntity.cardCount);
                outRGB += fresnel * reflectionRadiance;
            }
        }
    }

    outColor = vec4(outRGB, 1.0);
}
