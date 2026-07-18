#version 460
#extension GL_GOOGLE_include_directive : enable

// renderer::TransparentForwardPass's vertex stage -- mirrors ClusterRaster.vert's own vertex-
// pulling technique closely (decode straight from the compressed physical page pool, apply entity
// self-rotation then WPO sway), but reads renderer::TransparentClusterEntry (this pass's own small
// static list, see TransparentForwardPass.h's class comment) instead of ClusterCullMetadata, and
// outputs world-space position/normal (for real Lambertian+shadow shading in the fragment stage)
// instead of a Visibility Buffer ClusterID -- this pass shades directly, no deferred resolve.
//
// gl_InstanceIndex / gl_VertexIndex have the exact same meaning as ClusterRaster.vert's own
// (firstInstance == this entry's slot in g_Entries; the fixed-function stage already added
// VkDrawIndexedIndirectCommand::vertexOffset, written per-frame by TransparentClusterCompact.comp,
// to gl_VertexIndex before this shader runs).

#include "include/struct_custo.glsl"

#define COMPRESSED_POOL_SET 0
#define COMPRESSED_POOL_BINDING 1
#include "include/cluster_vertex_decode.glsl"
#include "include/wpo_deformation.glsl"
#include "include/enhanced_displacement.glsl"
#include "include/spline_deformation.glsl"
// Skeletal-animation feature: this entity's clusters never carry core::EntityFlags::
// IsSkeletallyAnimated (the one skinned entity, the procedural creature, is always fully opaque --
// see VulkanContext::BuildEntityData()'s own comment -- so its clusters never reach this
// TRANSPARENT-only forward pass), so ApplySkeletalSkinning() is never actually called here. Only
// included for the SKELETAL_MAX_DEVIATION constant displacement_bounds.glsl references
// unconditionally below (this file has no bone-matrices SSBO binding of its own).
#include "include/skeletal_animation.glsl"
#include "include/displacement_bounds.glsl"

// Mirrors renderer::TransparentClusterEntry (TransparentForwardPass.h) field-for-field.
struct TransparentClusterEntry {
    vec3 boundsMin; float maxWPOAmplitude;
    vec3 boundsMax; float _pad0;
    uint logicalPageID;
    uint indexCount;
    uint clusterID;
    uint entityID;
    uint materialID;
    uint maskTextureIndex;
    float _pad1;
    float _pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer TransparentClusterEntriesSSBO {
    TransparentClusterEntry entries[];
} g_Entries;

// binding 1 == g_CompressedClusterPool, declared by cluster_vertex_decode.glsl.

layout(std430, set = 0, binding = 2) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(std430, set = 0, binding = 3) readonly buffer EntityDataBuffer {
    EntityData entityData[];
};

layout(std140, set = 0, binding = 4) uniform WPOGlobalsUBO {
    float globalTime;
    float enhancedDisplacementDebugMultiplier;
    float splineDeformationDebugMultiplier;
    float _pad2;
} g_WPOGlobals;

layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view;
    mat4 proj;
    vec3 sunDirection; // Unused here (fragment-stage only) -- see TransparentForward.frag.
    float _pad0;
} g_ViewParams;

// Phase 1 (Nanite advanced): see ClusterRaster.vert's identical binding comment. Binding 18 --
// the first free slot past this pass's full 0-17 range (TransparentForward.frag's own Phase 3/
// MegaLights/World-Probe/reflection-trace bindings occupy 6-17, see TransparentForwardPass::Init's
// own binding layout).
layout(std430, set = 0, binding = 18) readonly buffer SplineControlPointsSSBO {
    SplineControlPoint splineControlPoints[SPLINE_CONTROL_POINT_COUNT];
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out uint outMaterialID;

void main() {
    TransparentClusterEntry entry = g_Entries.entries[gl_InstanceIndex];

    // See ClusterRaster.vert's identical comment: vertexOffset was already added to gl_VertexIndex
    // by the fixed-function stage from this frame's VkDrawIndexedIndirectCommand (written by
    // TransparentClusterCompact.comp), so it must be subtracted back off to recover the cluster-
    // local vertex slot the compressed page's SoA layout is indexed by.
    uint physicalPageIndex = uint(gl_VertexIndex) / CLUSTER_MAX_VERTICES; // vertexOffset is always physicalPageIndex * CLUSTER_MAX_VERTICES.
    uint localVertexIndex = uint(gl_VertexIndex) % CLUSTER_MAX_VERTICES;
    uint pageByteBase = physicalPageIndex * CLUSTER_PAGE_SIZE_BYTES;

    vec3 worldPos = DecodeClusterPosition(pageByteBase, localVertexIndex, entry.boundsMin, entry.boundsMax);
    vec3 normal = DecodeClusterNormal(pageByteBase, localVertexIndex);

    // Apply entity self-rotation -- see ClusterRaster.vert's identical comment.
    EntityData ed = entityData[entry.entityID];
    EntityTransform xform = entityTransforms[ed.meshID];
    mat3 rotation = mat3(xform.rotation);
    vec3 localPos = worldPos - xform.center;

    // Phase 1 (Nanite advanced): spline bend, applied in LOCAL space BEFORE the per-entity rigid
    // rotation -- see ClusterRaster.vert's identical comment.
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_SPLINE_DEFORMATION)) {
        localPos = mix(localPos, ApplySplineDeformation(localPos, splineControlPoints), g_WPOGlobals.splineDeformationDebugMultiplier);
    }

    worldPos = xform.translation + xform.center + rotation * localPos;
    normal = rotation * normal;

    worldPos = ApplyWPODeformation(worldPos, entry.clusterID, GetOriginalWPOAmplitude(entry.maxWPOAmplitude, ed.flags), g_WPOGlobals.globalTime);

    // Phase 1 (Nanite advanced): multi-octave enhanced displacement, applied ADDITIVELY right after
    // WPO sway -- see ClusterRaster.vert's identical comment.
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT)) {
        vec3 displaced = ApplyEnhancedDisplacement(worldPos, xform.center, entry.clusterID, g_WPOGlobals.globalTime);
        worldPos = worldPos + (displaced - worldPos) * g_WPOGlobals.enhancedDisplacementDebugMultiplier;
    }

    outWorldPos = worldPos;
    outNormal = normal;
    outMaterialID = entry.materialID;

    gl_Position = g_ViewParams.proj * g_ViewParams.view * vec4(worldPos, 1.0);
}
