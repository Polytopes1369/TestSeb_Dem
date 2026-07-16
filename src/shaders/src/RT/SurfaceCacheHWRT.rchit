#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing : require

// Part 2 deliverable: Closest-Hit Shader. Instead of evaluating the object's heavy material
// shader, uses the hit instance's ID + triangle ID to retrieve the Surface Cache Card associated
// with this entity/face and samples the atlas -- the whole point of Lumen-style Surface Cache hit
// lighting: a ray that hits geometry far from the camera gets a cheap, pre-baked lookup instead of
// re-shading full material complexity.

#include "include/mesh_sdf_trace.glsl"        // set 1: EntityInfo (firstCardIndex/cardCount) -- only its card-range fields are used here, its SDF sampler array is not.
#include "include/surface_cache_sampling.glsl" // set 2: card table + radiance atlas.

struct RayResult {
    vec3 color;
    float hit;
};
layout(location = 0) rayPayloadInEXT RayResult g_Payload;

// Combined Fallback Mesh vertex/index buffers -- the SAME buffers every BLAS in this TLAS was
// built directly against (renderer::SurfaceCachePass::GetVertexBuffer()/GetIndexBuffer(), see
// renderer::AccelerationStructure::BuildBLAS's own comment on why no separate geometry upload
// exists for ray tracing). std430-exact mirror of geometry::FallbackVertex (32 bytes: 3+3+2
// floats, no padding -- see ClusterFormat.h's own static_assert) using flat scalars for the same
// reason surface_cache_sampling.glsl's card struct does (avoid std430's vec3-alignment padding
// diverging from the tightly-packed C++ layout).
struct FallbackVertexGpu {
    float posX, posY, posZ;
    float normX, normY, normZ;
    float u, v;
};
layout(std430, set = 3, binding = 0) readonly buffer FallbackVertexBuffer {
    FallbackVertexGpu g_Vertices[];
};
layout(std430, set = 3, binding = 1) readonly buffer FallbackIndexBuffer {
    uint g_Indices[];
};
// std430-exact mirror of renderer::SurfaceCachePass::EntityDrawRange (vertexOffset/firstIndex/
// indexCount, 12 bytes in C++ but padded to 16 here for std430's own array-stride rounding --
// the C++-side upload zero-fills the pad field, see SurfaceCacheRayTracingPass::Init()).
struct EntityDrawRangeGpu {
    int vertexOffset;
    uint firstIndex;
    uint indexCount;
    uint _pad;
};
layout(std430, set = 3, binding = 2) readonly buffer EntityDrawRangeBuffer {
    EntityDrawRangeGpu g_DrawRanges[];
};

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

void main() {
    // gl_InstanceCustomIndexEXT == the dense traced-entity index (renderer::SurfaceCacheTraceContext's
    // own array index, 0..entityCount), set as VkAccelerationStructureInstanceKHR::
    // instanceCustomIndex per instance at TLAS-build time -- see
    // renderer::SurfaceCacheRayTracingPass::Init(). This is deliberately NOT the sparse original
    // entityID (see GlobalSDFPass::GetTracedEntityInfos()'s own comment on why those differ), so
    // it indexes g_Entities/g_DrawRanges directly with no further lookup.
    uint entityIndex = gl_InstanceCustomIndexEXT;
    EntityDrawRangeGpu range = g_DrawRanges[entityIndex];

    uint triangleFirstIndex = range.firstIndex + gl_PrimitiveID * 3u;
    uint i0 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 0u];
    uint i1 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 1u];
    uint i2 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 2u];

    vec3 p0 = FetchPosition(i0);
    vec3 p1 = FetchPosition(i1);
    vec3 p2 = FetchPosition(i2);

    // Geometric (flat) face normal from the 3 corner positions -- more robust than an
    // interpolated per-vertex normal for a heavily QEM-simplified Fallback Mesh (see
    // geometry::FallbackMeshBuilder.h's own header comment on how aggressively it compacts), and
    // all SelectBestCard needs is "which face direction does this triangle roughly point along."
    vec3 faceNormal = normalize(cross(p1 - p0, p2 - p0));

    // Object space == world space == local space here: every TLAS instance uses an identity
    // transform (this codebase's entities carry no runtime transform -- see
    // SurfaceCacheCapture.vert's own comment), so the ray's object-space hit position is exactly
    // the local-space position ComputeCardLocalUV (surface_cache_sampling.glsl) expects.
    vec3 localHitPos = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;

    EntityInfo entity = g_Entities[entityIndex];

    g_Payload.color = SampleCardRadiance(localHitPos, faceNormal, entity.firstCardIndex, entity.cardCount);
    g_Payload.hit = 1.0;
}
