#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// Phase 7c (UE5.8 parity roadmap, water/erosion): forward-shaded water material fragment shader
// for renderer::WaterForwardPass. Modeled on Tessellation.frag's reflection-trace structure
// (front-layer GGX-VNDF specular against the shared Surface Cache trace scene, identical math),
// but with NO diffuse/shadowed/MegaLights term (water has none -- see WaterForwardPass' own class
// comment) and a real REFRACTION term neither hero nor glass needed: samples g_BackgroundSnapshot
// (a frozen copy of the already-composited frame, blitted by WaterForwardPass::RecordDraw BEFORE
// this shader ever runs -- see that pass' own header comment for why a live read of the same image
// would be undefined behavior) through a wave-perturbed UV offset, tinted by Beer-Lambert depth
// absorption using terrain_noise.glsl's own SampleTerrainHeight as the seabed depth reference (the
// same function the terrain itself is generated from -- always exactly consistent with the real
// rasterized seabed, no separate depth buffer needed).
//
// --- Compositing ---
// blendEnable=false (see WaterForwardPass' own pipeline state) -- this shader composes the FINAL
// pixel color itself: mix() between the refracted/tinted background and the reflected radiance,
// weighted by the Fresnel term (a wave-normal-dependent lerp is the standard real-time water
// technique -- near-vertical view angles read mostly as refraction/transparency, grazing angles
// read mostly as reflection, exactly like real water). outColor.a is always 1.0 (irrelevant, no
// blending reads it).

#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"
#include "include/material_params.glsl"
#include "include/terrain_noise.glsl" // SampleTerrainHeight -- the real seabed depth reference.
#include "include/water_params.glsl"  // kWaterLevel
#include "include/foam_generation.glsl" // Wave 2 (UE5.8 water/foam parity) -- declares binding 7 (FoamUBO).
// Rivers/waterfalls feature: this pass is the SHARED draw for both the flat lake plane and the
// river/waterfall ribbon (see geom_river.comp's own header comment for why they share one
// renderer::WaterForwardPass draw call instead of a dedicated river pass/entity) -- river_spline.glsl
// lets this shader classify "am I lake or river/waterfall this fragment" and derive a flow
// direction/speed purely from world position, since geometry::FallbackVertex carries no spare
// per-vertex channel (uv2) through the Fallback Mesh proxy this pass actually draws from.
#include "include/river_spline.glsl"

// Single-element buffer -- this pass only ever shades ONE fixed material (see
// renderer::WaterForwardPass.cpp's own `waterMaterial` parameter comment), same convention
// renderer::TessellationPass established.
layout(std430, set = 0, binding = 0) readonly buffer MaterialParamsSSBO {
    MaterialParams mat;
} g_MaterialParams;

// binding 1 == g_EntityTransforms, declared by WaterForward.vert only -- this fragment shader
// never needs it, the vertex shader already resolved world position.

// HWRT resources (own set 0 bindings 3-5, NOT part of SurfaceCacheTraceContext's set 1/2 --
// mirrors Tessellation.frag's own equivalent bindings).
layout(set = 0, binding = 2) uniform accelerationStructureEXT g_TLAS;
struct FallbackVertexGpu { float posX, posY, posZ; float normX, normY, normZ; float u, v; };
layout(std430, set = 0, binding = 3) readonly buffer FallbackVertexBuffer { FallbackVertexGpu g_Vertices[]; };
layout(std430, set = 0, binding = 4) readonly buffer FallbackIndexBuffer { uint g_Indices[]; };
struct EntityDrawRangeGpu { int vertexOffset; uint firstIndex; uint indexCount; uint _pad; };
layout(std430, set = 0, binding = 5) readonly buffer EntityDrawRangeBuffer { EntityDrawRangeGpu g_DrawRanges[]; };

layout(set = 0, binding = 6) uniform sampler2D g_BackgroundSnapshot;

#include "include/mesh_sdf_trace.glsl"        // set 1
#include "include/surface_cache_sampling.glsl" // set 2

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;

layout(push_constant) uniform WaterForwardConstants {
    mat4 viewProj;
    float cameraPositionWorldX, cameraPositionWorldY, cameraPositionWorldZ;
    float _pad0;
    uint entityID;
    uint traceMode;
    uint frameIndex;
    uint entityCount;
    float viewportWidth, viewportHeight;
    float timeSeconds;
    float _pad1;
} pc;

layout(location = 0) out vec4 outColor;

// --- Waves: per-fragment normal perturbation only, NO vertex displacement (see this file's own
// header comment) -- same analytic finite-difference technique as geom_terrain.comp's own normal
// computation, applied to a temporal single-octave noise field instead of the real terrain relief
// (a gentle swell doesn't need multi-octave detail -- a quarter of the cost for a fully sufficient
// "soft rolling water" look). Kept inline here, single consumer, no second GPU user unlike
// terrain_noise.glsl's own SampleTerrainHeight.
const float kWaveFrequency = 0.6;
const float kWaveAmplitude = 0.15;
const float kWaveSpeed = 0.35;

float WaveHeight(vec2 worldXZ, float timeSeconds) {
    return ValueNoise3D(vec3(worldXZ.x, timeSeconds * kWaveSpeed, worldXZ.y) * kWaveFrequency) * 2.0 - 1.0;
}

vec3 ComputeWaveNormal(vec2 worldXZ, float timeSeconds) {
    float eps = 0.3;
    float hL = WaveHeight(worldXZ - vec2(eps, 0.0), timeSeconds);
    float hR = WaveHeight(worldXZ + vec2(eps, 0.0), timeSeconds);
    float hD = WaveHeight(worldXZ - vec2(0.0, eps), timeSeconds);
    float hU = WaveHeight(worldXZ + vec2(0.0, eps), timeSeconds);
    vec3 tangentX = vec3(2.0 * eps, (hR - hL) * kWaveAmplitude, 0.0);
    vec3 tangentZ = vec3(0.0, (hU - hD) * kWaveAmplitude, 2.0 * eps);
    return normalize(cross(tangentZ, tangentX));
}

// Refraction / absorption tuning -- world-unit depth scale, screen-space UV-offset strength.
const float kRefractionStrength = 0.03;
const float kAbsorptionCoefficient = 1.2;

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

// Nth copy of this codebase's established inline-rayQuery HWRT trace (see Tessellation.frag's
// own header comment for the full list of existing copies and why a further one here follows the
// existing convention rather than refactoring already-working files).
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

void main() {
    // --- Rivers/waterfalls feature: classify this fragment against river_spline.glsl's authored
    // path -- riverMask is 0 on the open lake (unchanged behavior below) and ramps to 1 on/near the
    // river or waterfall channel; riverTangentXZ is the local downstream direction; riverSlope is
    // the local downhill steepness (world-height-drop per world-XZ-arc-unit), near-vertical at the
    // one waterfall segment (see river_spline.glsl's own kRiverControlHeight comment) -- used below
    // to scale both the flow-scroll speed and the whitewater/foam brightening. ---
    float riverT, riverSurfaceHeight, riverDistance;
    vec2 riverClosestXZ, riverTangentXZ;
    ClosestPointOnRiverPolyline(inWorldPos.xz, riverT, riverClosestXZ, riverSurfaceHeight, riverTangentXZ, riverDistance);
    float riverMask = 1.0 - smoothstep(kRiverBandInnerHalfWidth, kRiverBandOuterHalfWidth, riverDistance);
    float riverSlope = riverMask > 0.0 ? RiverSlopeAtT(riverT) : 0.0;

    // Flow-mapped UV scroll: shifts the sampling coordinate upstream over time (so the noise
    // pattern itself appears to travel downstream, along riverTangentXZ) -- scaled by riverMask so
    // this is an exact no-op on the open lake (mask == 0, sampleXZ == inWorldPos.xz unchanged).
    // kRiverBaseFlowSpeed/kRiverSlopeSpeedGain: a gentle river reach still visibly drifts; the
    // waterfall segment's own much larger riverSlope drives the speed far higher, exactly the
    // "steeper == faster visual flow" requirement this feature targets.
    const float kRiverBaseFlowSpeed = 0.6;
    const float kRiverSlopeSpeedGain = 2.5;
    float flowSpeed = kRiverBaseFlowSpeed + riverSlope * kRiverSlopeSpeedGain;
    vec2 flowShiftedXZ = inWorldPos.xz - riverTangentXZ * (pc.timeSeconds * flowSpeed * riverMask);

    // The wave-perturbed normal IS this surface's shading normal -- a flat, wave-free water plane
    // has inWorldNormal == (0,1,0) exactly, and ComputeWaveNormal reduces to the same (0,1,0) at
    // zero amplitude (its own construction, mirroring geom_terrain.comp's identical guarantee).
    vec3 n = ComputeWaveNormal(flowShiftedXZ, pc.timeSeconds);
    // Blends in a higher-frequency, faster-evolving second wave sample where the river is both
    // present AND steep (rapids/waterfall) -- a cheap stand-in for real turbulence, giving the
    // waterfall segment a visibly choppier normal than the calm upstream reach or the lake.
    if (riverMask > 0.0 && riverSlope > 0.0) {
        vec3 turbulentNormal = ComputeWaveNormal(flowShiftedXZ * 3.0, pc.timeSeconds * 1.7);
        n = normalize(mix(n, turbulentNormal, riverMask * clamp(riverSlope * 0.6, 0.0, 1.0)));
    }
    MaterialParams mat = g_MaterialParams.mat;

    // --- Refraction: sample the frozen background snapshot through a wave-perturbed UV offset,
    // tinted by Beer-Lambert depth absorption (real seabed depth from terrain_noise.glsl's own
    // SampleTerrainHeight -- always exactly consistent with the rasterized terrain). Substrate:
    // mat.base.diffuseAlbedo is this recipe's deep blue-teal tint (see
    // renderer::GenerateShowcaseMaterialTable's own kWaterMaterialID comment), mat.alpha is
    // repurposed as the maximum absorption blend strength (NOT a fixed-function blend alpha --
    // this shader composes manually, see this file's own header comment). ---
    vec2 screenUV = gl_FragCoord.xy / vec2(pc.viewportWidth, pc.viewportHeight);
    // Depth reference is THIS fragment's own rasterized surface height (inWorldPos.y), not the
    // flat-lake-only kWaterLevel constant: exactly equal to kWaterLevel on the lake (a flat plane,
    // so this is a no-op there), but correctly follows the river ribbon's own authored, non-flat
    // surface height upstream of the lake (see river_spline.glsl's own kRiverControlHeight comment
    // for why a river's water surface is not a single fixed Y the way the lake's is).
    float waterDepth = max(inWorldPos.y - SampleTerrainHeight(inWorldPos.xz), 0.0);
    // Attenuate the UV offset near the shore (shallow water) so the refraction distortion doesn't
    // visibly displace the shoreline itself -- same smoothstep-taper idiom terrain_shading.glsl's
    // own beach band already uses.
    vec2 refractOffset = n.xz * kRefractionStrength * smoothstep(0.0, 0.3, waterDepth);
    vec3 backgroundColor = texture(g_BackgroundSnapshot, clamp(screenUV + refractOffset, 0.0, 1.0)).rgb;
    float absorption = (1.0 - exp(-waterDepth * kAbsorptionCoefficient)) * mat.alpha;
    vec3 refractedTinted = mix(backgroundColor, mat.base.diffuseAlbedo, absorption);

    // --- Front-layer specular reflection (Lumen-style, single GGX-VNDF sample, identical
    // technique to Tessellation.frag's own optional-reflection block) ---
    vec3 cameraPositionWorld = vec3(pc.cameraPositionWorldX, pc.cameraPositionWorldY, pc.cameraPositionWorldZ);
    vec3 viewDir = normalize(cameraPositionWorld - inWorldPos);
    float NdotV = max(dot(n, viewDir), 0.0);
    // Dielectric water: F0 ~= 0.02 (IOR 1.33).
    vec3 F0 = vec3(0.02);
    vec3 fresnel = F_Schlick(F0, NdotV);

    vec3 tangent, bitangent;
    BuildTangentBasis(n, tangent, bitangent);
    vec3 viewDirTangentSpace = vec3(dot(viewDir, tangent), dot(viewDir, bitangent), dot(viewDir, n));

    uint64_t pixelSeed = uint64_t(uint(gl_FragCoord.x)) | (uint64_t(uint(gl_FragCoord.y)) << 32);
    vec2 xi = vec2(
        hashFloat(pixelSeed, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0x517CC1B7UL),
        hashFloat(pixelSeed, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0xBF58476D1CE4E5B9UL));

    vec3 halfVectorTangentSpace = SampleGGXVNDF(xi, viewDirTangentSpace, mat.base.roughness);
    vec3 halfVectorWorld = normalize(
        tangent * halfVectorTangentSpace.x + bitangent * halfVectorTangentSpace.y + n * halfVectorTangentSpace.z);
    vec3 rayDir = reflect(-viewDir, halfVectorWorld);

    vec3 reflectionRadiance = vec3(0.0);
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
    // Grazing-angle GGX-VNDF edge case and missed reflection rays both leave reflectionRadiance at
    // 0.0 -- same handling as Tessellation.frag's own (no sky/ambient fallback exists anywhere
    // in this codebase; water inherits that same accepted limitation, not a regression).

    // --- Composite: Fresnel-weighted lerp between refraction and reflection (see this file's own
    // header comment) -- NOT an additive blend, since this shader writes the final opaque pixel
    // directly (blendEnable=false). ---
    float fresnelScalar = clamp(fresnel.r, 0.0, 1.0); // F0 is achromatic, so r==g==b already.
    vec3 outRGB = mix(refractedTinted, reflectionRadiance, fresnelScalar);

    // --- Rivers/waterfalls feature: whitewater/foam -- brightens toward near-white proportionally
    // to riverMask (am I on the channel at all) times riverSlope (how steep, unsigned m/m), so the
    // calm upstream reach stays a normal tinted-water look while the waterfall segment and any
    // sharp rapids read as visibly foamy, without needing a separate foam texture/material. ---
    float whitewater = riverMask * clamp(riverSlope * 0.5, 0.0, 1.0);
    outRGB = mix(outRGB, vec3(0.90, 0.95, 1.0), whitewater * 0.6);

    // --- Wave 2 (UE5.8 water/foam parity): wave-breaking foam on the OPEN LAKE -- distinct from the
    // river/waterfall whitewater above (which is keyed off the authored river channel, not actual
    // wave steepness), this is real DetectWaveBreaking()/GenerateFoamPattern() foam driven by the
    // swell noise's own height derivative (steep/choppy patches break), gated by real shore
    // proximity (this shader's own `waterDepth`, not the synthetic swell noise -- foam reads
    // clearly near the shore/shallow basin, fades out over open deep water, matching real
    // shallow-water wave breaking). Fills a genuine gap the river-only whitewater above never
    // covered: previously the open lake had no foam at all. ---
    {
        const float kDerivativeEps = 0.3;
        vec2 waveHeightDerivative = vec2(
            WaveHeight(flowShiftedXZ + vec2(kDerivativeEps, 0.0), pc.timeSeconds) - WaveHeight(flowShiftedXZ - vec2(kDerivativeEps, 0.0), pc.timeSeconds),
            WaveHeight(flowShiftedXZ + vec2(0.0, kDerivativeEps), pc.timeSeconds) - WaveHeight(flowShiftedXZ - vec2(0.0, kDerivativeEps), pc.timeSeconds)
        ) * kWaveAmplitude;
        float waveHeightNow = WaveHeight(flowShiftedXZ, pc.timeSeconds) * kWaveAmplitude;
        float breakingAmount = DetectWaveBreaking(waveHeightDerivative, waveHeightNow, g_Foam.breakingWaveThreshold.x);
        // Shallow-water fade: only the real seabed-relative depth (waterDepth, computed above for
        // refraction) can tell shore from open basin -- the synthetic swell noise alone cannot.
        breakingAmount *= 1.0 - smoothstep(0.0, 1.5, waterDepth);
        vec4 lakeFoam = GenerateFoamPattern(inWorldPos, pc.timeSeconds, breakingAmount);
        outRGB = BlendFoam(vec4(outRGB, 1.0), lakeFoam, 1.0).rgb;
    }

    outColor = vec4(outRGB, 1.0);
}
