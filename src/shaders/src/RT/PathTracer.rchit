#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

// UE5.8 rendering-parity gap G10b -- reference Path Tracer closest-hit shader (DEBUG-only, see
// PathTracer.rgen's header). Resolves a scene-ray hit into the surface data the ray-gen shader's
// megakernel needs: world-space hit position, world-space geometric face normal, and this entity's
// materialID. All light-transport / BSDF / NEE logic lives in the ray-gen shader -- this shader is
// purely hit-data resolution, mirroring how SurfaceCacheHWRT.rchit resolves a hit into a Surface
// Cache lookup.

// Fallback Mesh vertex/index/draw-range buffers (the SAME geometry every BLAS in the TLAS was built
// against) at set 0, bindings 2/3/4 -- shared include, identical convention to SurfaceCacheHWRT.rchit
// and ReflectionTrace.comp.
#define FALLBACK_GEOMETRY_SET 0
#define FALLBACK_GEOMETRY_BASE_BINDING 2
#include "include/fallback_geometry.glsl"

// Per-traced-entity materialID, indexed by the dense traced-entity index (== gl_InstanceCustomIndexEXT,
// the TLAS instanceCustomIndex renderer::SurfaceCacheRayTracingPass assigns). Built once by
// renderer::PathTracerPass::Init from core::EntityData (entityDataCPU[tracedEntity.entityID].materialID)
// so this shader needs no entityID->materialID indirection at trace time.
layout(std430, set = 0, binding = 5) readonly buffer TracedMaterialIDBuffer {
    uint g_TracedMaterialID[];
};

// Payload -- byte-layout must match PathTracer.rgen / PathTracer.rmiss exactly.
struct PTPayload {
    vec3 hitPos;
    vec3 hitNormal;
    uint materialID;
    float hitT;
};
layout(location = 0) rayPayloadInEXT PTPayload g_Payload;

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

void main() {
    uint entityIndex = gl_InstanceCustomIndexEXT;
    EntityDrawRangeGpu range = g_DrawRanges[entityIndex];

    uint triangleFirstIndex = range.firstIndex + uint(gl_PrimitiveID) * 3u;
    uint i0 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 0u];
    uint i1 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 1u];
    uint i2 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 2u];

    vec3 p0 = FetchPosition(i0);
    vec3 p1 = FetchPosition(i1);
    vec3 p2 = FetchPosition(i2);

    // Geometric (flat) face normal in object space -- robust for the heavily QEM-simplified Fallback
    // Mesh (same choice SurfaceCacheHWRT.rchit makes). Transformed to world space by the instance's
    // rotation so it stays correct even when entity self-rotation is enabled (TLAS refit); for the
    // default identity-transform case mat3(gl_ObjectToWorldEXT) is identity, a no-op.
    vec3 faceNormalObject = normalize(cross(p1 - p0, p2 - p0));
    vec3 faceNormalWorld = normalize(mat3(gl_ObjectToWorldEXT) * faceNormalObject);

    g_Payload.hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    g_Payload.hitNormal = faceNormalWorld;
    g_Payload.materialID = g_TracedMaterialID[entityIndex];
    g_Payload.hitT = gl_HitTEXT;
}
