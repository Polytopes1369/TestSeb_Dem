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
    float _pad0;
    float _pad1;
    float _pad2;
} g_WPOGlobals;

layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view;
    mat4 proj;
    vec3 sunDirection; // Unused here (fragment-stage only) -- see TransparentForward.frag.
    float _pad0;
} g_ViewParams;

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
    worldPos = xform.center + rotation * (worldPos - xform.center);
    normal = rotation * normal;

    worldPos = ApplyWPODeformation(worldPos, entry.clusterID, entry.maxWPOAmplitude, g_WPOGlobals.globalTime);

    outWorldPos = worldPos;
    outNormal = normal;
    outMaterialID = entry.materialID;

    gl_Position = g_ViewParams.proj * g_ViewParams.view * vec4(worldPos, 1.0);
}
