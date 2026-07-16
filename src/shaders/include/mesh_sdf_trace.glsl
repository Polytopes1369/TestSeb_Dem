#ifndef MESH_SDF_TRACE_GLSL
#define MESH_SDF_TRACE_GLSL

#include "include/math_utils.glsl"

// Software ray tracing (SWRT) primitive: traces a ray against every traced entity's per-object
// Mesh SDF (renderer::GlobalSDFPass's own EntitySDF images, reused unmodified via
// GetTracedEntityInfos() -- see renderer::SurfaceCacheTraceContext for how this file's binding
// set is populated) by sphere tracing, keeping the globally-nearest hit across every entity. No
// BVH: a linear scan over a demoscene-scale entity count (see kMaxTracedEntities) with a cheap
// ray-AABB pretest before ever sphere-tracing an entity's own SDF -- consistent with this
// codebase's existing "software" simplicity bias (geometry::SurfaceCacheAtlasAllocator's own
// class comment makes the same call for a comparably small working set).
//
// Binding convention every including shader must reserve -- descriptor set 1 is "mesh SDF trace
// scene":
//   set 1, binding 0: readonly EntityInfoBuffer SSBO (see EntityInfo below), entityCount entries.
//   set 1, binding 1: sampler3D g_EntitySDF[kMaxTracedEntities] (unused slots >= entityCount
//                      point at a 1x1x1 dummy volume sampled kFarDistance everywhere -- see
//                      renderer::SurfaceCacheTraceContext::Init()).
//
// Local-space == world-space throughout this file: this codebase's entities carry no runtime
// transform (see SurfaceCacheCapture.vert's own comment), so a ray's origin/direction as supplied
// by the caller is used directly against every entity's own local-space SDF volume, with no
// per-entity inverse-transform step anywhere below.

const uint kMaxTracedEntities = 64u;
const float kFarDistance = 1.0e4;
const float kSphereTraceEpsilon = 1.0e-3;
const uint kSphereTraceMaxSteps = 96u;
const float kSphereTraceMinStep = 1.0e-3;

// One traced entity's Mesh SDF volume descriptor + Surface Cache card range. std430-exact mirror
// of a flat scalar layout (see surface_cache_sampling.glsl's own struct-packing comment for why
// this matters): 3 floats (volumeMin) + 1 float (voxelSize) + 4 uints = 32 bytes, no vec3-stride
// padding surprises.
struct EntityInfo {
    float volumeMinX, volumeMinY, volumeMinZ;
    float voxelSize;
    uint resolution;
    uint entityID;          // Debug/traceback only -- g_Cards is looked up via firstCardIndex, not entityID.
    uint firstCardIndex;    // Offset into EntityCardIndexBuffer (surface_cache_sampling.glsl).
    uint cardCount;
};

layout(std430, set = 1, binding = 0) readonly buffer EntityInfoBuffer {
    EntityInfo g_Entities[];
};
layout(set = 1, binding = 1) uniform sampler3D g_EntitySDF[kMaxTracedEntities];

vec3 EntityVolumeMin(EntityInfo e) { return vec3(e.volumeMinX, e.volumeMinY, e.volumeMinZ); }
vec3 EntityVolumeMax(EntityInfo e) { return EntityVolumeMin(e) + vec3(float(e.resolution) * e.voxelSize); }

// Trilinear sample of traced entity `entityIndex`'s Mesh SDF at local-space position `localPos`,
// matching geometry::SampleMeshSDF's voxel-center convention (see GlobalSDFComposite.comp's own
// derivation comment for why no extra +/-0.5 term belongs in the uvw computation below -- hardware
// trilinear filtering on normalized coordinates already reproduces it exactly).
float SampleEntitySDF(uint entityIndex, EntityInfo e, vec3 localPos) {
    vec3 uvw = (localPos - EntityVolumeMin(e)) / (e.voxelSize * float(e.resolution));
    return texture(g_EntitySDF[entityIndex], uvw).r;
}

// Slab test against [EntityVolumeMin(e), EntityVolumeMax(e)); returns false (tEnter/tExitOut
// untouched) if the ray misses the box entirely or the intersection interval does not overlap
// [0, tMax] (tMax lets the caller reject an entity whose whole AABB is already farther away than
// the closest hit found so far -- see TraceMeshSDFScene).
bool IntersectEntityAABB(vec3 rayOrigin, vec3 rayDir, EntityInfo e, float tMax, out float tEnter, out float tExitOut) {
    vec3 boundsMin = EntityVolumeMin(e);
    vec3 boundsMax = EntityVolumeMax(e);
    vec3 invDir = 1.0 / rayDir;
    vec3 t0 = (boundsMin - rayOrigin) * invDir;
    vec3 t1 = (boundsMax - rayOrigin) * invDir;
    vec3 tSmall = min(t0, t1);
    vec3 tBig = max(t0, t1);
    tEnter = max(max(tSmall.x, tSmall.y), max(tSmall.z, 0.0));
    tExitOut = min(min(tBig.x, tBig.y), min(tBig.z, tMax));
    return tEnter <= tExitOut;
}

// Sphere traces traced entity `entityIndex` over ray segment [tEnter, tExit] (already clipped to
// its own AABB by IntersectEntityAABB). Returns true and the hit distance in `outT` on a surface
// crossing (SDF magnitude <= kSphereTraceEpsilon), false (segment exhausted / step budget
// exhausted) otherwise -- standard sphere tracing: the SDF's own value is always a safe lower
// bound on distance to the nearest surface in ANY direction, so marching exactly that far can
// never step past a surface unnoticed. kSphereTraceMinStep prevents stalling very near the zero
// crossing when the field's local gradient magnitude is < 1 (an expected side effect of the
// narrow-band clamped/BC4-style compressed storage -- see geometry::MeshSDFGenerator.h).
bool SphereTraceEntity(uint entityIndex, EntityInfo e, vec3 rayOrigin, vec3 rayDir, float tEnter, float tExit, out float outT) {
    float t = tEnter;
    for (uint step = 0u; step < kSphereTraceMaxSteps && t <= tExit; ++step) {
        vec3 pos = rayOrigin + rayDir * t;
        float dist = SampleEntitySDF(entityIndex, e, pos);
        if (dist <= kSphereTraceEpsilon) {
            outT = t;
            return true;
        }
        t += max(dist, kSphereTraceMinStep);
    }
    return false;
}

// Central-difference SDF gradient at `localPos`, normalized -- the outward surface normal
// SelectBestCard (surface_cache_sampling.glsl) needs to pick this entity's best-fit card.
vec3 EntitySDFNormal(uint entityIndex, EntityInfo e, vec3 localPos) {
    float h = e.voxelSize * 0.5;
    float dx = SampleEntitySDF(entityIndex, e, localPos + vec3(h, 0.0, 0.0)) - SampleEntitySDF(entityIndex, e, localPos - vec3(h, 0.0, 0.0));
    float dy = SampleEntitySDF(entityIndex, e, localPos + vec3(0.0, h, 0.0)) - SampleEntitySDF(entityIndex, e, localPos - vec3(0.0, h, 0.0));
    float dz = SampleEntitySDF(entityIndex, e, localPos + vec3(0.0, 0.0, h)) - SampleEntitySDF(entityIndex, e, localPos - vec3(0.0, 0.0, h));
    return SafeNormalize(vec3(dx, dy, dz));
}

// The core SWRT primitive: linearly scans every traced entity in [0, entityCount) (see this
// file's own class comment for why no BVH), ray-AABB-prunes each one, and sphere traces the
// survivors, keeping the globally nearest hit. Returns true (with outEntityIndex/outLocalPos/
// outLocalNormal filled in) on a hit within [0, tMax], false otherwise. `entityCount` must be a
// dynamically uniform value (a push constant, in every consumer of this file) -- see this file's
// header comment on g_EntitySDF's array indexing requirement.
bool TraceMeshSDFScene(vec3 rayOrigin, vec3 rayDir, float tMax, uint entityCount,
    out uint outEntityIndex, out vec3 outLocalPos, out vec3 outLocalNormal) {
    float bestT = tMax;
    int bestEntity = -1;

    for (uint i = 0u; i < entityCount; ++i) {
        EntityInfo e = g_Entities[i];
        float tEnter, tExit;
        if (!IntersectEntityAABB(rayOrigin, rayDir, e, bestT, tEnter, tExit)) {
            continue;
        }
        float hitT;
        if (SphereTraceEntity(i, e, rayOrigin, rayDir, tEnter, tExit, hitT) && hitT < bestT) {
            bestT = hitT;
            bestEntity = int(i);
        }
    }

    if (bestEntity < 0) {
        return false;
    }

    outEntityIndex = uint(bestEntity);
    outLocalPos = rayOrigin + rayDir * bestT;
    outLocalNormal = EntitySDFNormal(outEntityIndex, g_Entities[outEntityIndex], outLocalPos);
    return true;
}

#endif
