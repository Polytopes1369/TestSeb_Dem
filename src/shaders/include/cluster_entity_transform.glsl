#ifndef CLUSTER_ENTITY_TRANSFORM_GLSL
#define CLUSTER_ENTITY_TRANSFORM_GLSL

#include "struct_custo.glsl"

// Per-cluster bounding volumes (AABB, bounding sphere, normal cone) are baked once at cluster-DAG-
// build time in the entity's rest pose -- they never account for the per-entity self-rotation that
// ClusterRaster.vert/draw.vert apply to the actual vertices every frame (see those shaders' own
// "worldPos = xform.center + rotation * (worldPos - xform.center)" formula). Any culling/LOD stage
// that tests rest-pose bounds directly against the current camera therefore drifts out of sync with
// where the geometry is actually drawn as soon as the entity has rotated away from its rest pose --
// on curved/silhouette-heavy clusters this misclassifies visible clusters as backfacing or out of
// frustum, producing holes in still-visible geometry (and, via the LOD screen-error test's now-stale
// projected size, inconsistent parent/child LOD selection -- z-fighting). The three helpers below
// re-derive each bounding volume in the entity's CURRENT orientation, using the exact same formula
// as the vertex shaders, so every downstream test stays consistent with what is actually rasterized.

// Rotates a rest-pose world-space point about the entity's own pivot (xform.center) and applies
// its additional world-space translation (see struct_custo.glsl's EntityTransform comment) --
// identical to ClusterRaster.vert's per-vertex transform.
vec3 TransformClusterCenterByEntity(vec3 restPoseCenter, EntityTransform xform) {
    mat3 rotation = mat3(xform.rotation);
    return xform.translation + xform.center + rotation * (restPoseCenter - xform.center);
}

// Rotates a direction (no pivot/translation involved): used for the cluster's normal cone axis.
vec3 TransformClusterDirectionByEntity(vec3 restPoseDirection, EntityTransform xform) {
    return mat3(xform.rotation) * restPoseDirection;
}

// Re-fits an axis-aligned bounding box after rotating it about the entity's pivot: transforms all 8
// corners through the same rest-pose-to-current formula and takes the new min/max. A pure rotation
// preserves each corner's distance from the pivot, so the true rotated volume is always fully
// contained in this refit box -- it can only be conservatively wider than the exact rotated
// silhouette, never narrower, which keeps both frustum and occlusion culling safe.
void TransformClusterBoundsByEntity(inout vec3 boundsMin, inout vec3 boundsMax, EntityTransform xform) {
    vec3 corners[8] = vec3[8](
        vec3(boundsMin.x, boundsMin.y, boundsMin.z),
        vec3(boundsMax.x, boundsMin.y, boundsMin.z),
        vec3(boundsMin.x, boundsMax.y, boundsMin.z),
        vec3(boundsMax.x, boundsMax.y, boundsMin.z),
        vec3(boundsMin.x, boundsMin.y, boundsMax.z),
        vec3(boundsMax.x, boundsMin.y, boundsMax.z),
        vec3(boundsMin.x, boundsMax.y, boundsMax.z),
        vec3(boundsMax.x, boundsMax.y, boundsMax.z));

    vec3 newMin = TransformClusterCenterByEntity(corners[0], xform);
    vec3 newMax = newMin;
    for (int i = 1; i < 8; ++i) {
        vec3 transformed = TransformClusterCenterByEntity(corners[i], xform);
        newMin = min(newMin, transformed);
        newMax = max(newMax, transformed);
    }
    boundsMin = newMin;
    boundsMax = newMax;
}

#endif // CLUSTER_ENTITY_TRANSFORM_GLSL
