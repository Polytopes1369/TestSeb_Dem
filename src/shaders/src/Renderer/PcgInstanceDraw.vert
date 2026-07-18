#version 460
#extension GL_GOOGLE_include_directive : enable

// PCG Instance Draw Path (Phase 0.2, UE5.8-parity PCG roadmap): renders a GPU-driven pool of
// PCG-spawned instances -- real baked Nanite cluster geometry (rock/bush/tree/debris archetypes
// today, any future baked meshID tomorrow), NOT a billboard quad (unlike ParticleRender.vert) --
// through the SAME cluster-culling/indirect-draw contract every fixed scene entity uses
// (renderer::ClusterCullingPass -- see that class' own header comment for the exact
// UploadClusterMetadata -> RecordClear -> RecordCull -> vkCmdDrawIndexedIndirectCount sequence),
// but with a genuinely PER-INSTANCE world transform instead of the fixed scene's one-slot-per-meshID
// EntityTransform lookup. renderer::PcgInstanceDrawPass owns a completely separate GPU-side
// instance pool from both core::InstanceRegistry (CPU-side EntityData) and the fixed
// EntityTransformBuffer ClusterRaster.vert reads -- see that class' own header comment.
//
// gl_InstanceIndex re-indexes ClusterCullMetadataSSBO exactly like ClusterRaster.vert's own
// gl_InstanceIndex convention (every surviving candidate's VkDrawIndexedIndirectCommand::
// firstInstance carries its own slot in this same array -- see ClusterFrustumCull.comp) -- EXCEPT
// this pipeline repurposes ClusterCullMetadata::entityID to mean "this candidate cluster's owning
// PCG instance slot index" (into PcgInstanceDataSSBO below), NOT a real scene meshID. This is safe
// ONLY because this pipeline's output is consumed exclusively by PcgInstanceDraw.frag below, never
// by the shared opaque Nanite resolve path (renderer::ClusterResolvePass), which is the only other
// consumer that assigns entityID its usual meaning. ClusterCullMetadata::materialID keeps its
// normal meaning unchanged (indexes MaterialParamsSSBO, binding 4 in the fragment shader).
//
// IMPORTANT (cluster bounds: decode vs culling -- see project memory
// project_cluster_bounds_decode_vs_culling.md): ClusterCullMetadata::boundsMin/boundsMax here hold
// this candidate's WORLD-space AABB (renderer::PcgInstanceDrawPass::UploadInstances transforms each
// instance's local AABB corners by that instance's own world transform before upload) -- required
// for renderer::ClusterCullingPass's frustum test (ClusterFrustumCull.comp) to be correct once a
// non-identity per-instance transform is involved. Vertex DEQUANTIZATION therefore CANNOT reuse
// those same fields (DecodeClusterPosition needs the ORIGINAL LOCAL-space AABB the CPU quantized
// each vertex against, geometry::ClusterIndexEntry::boundsMin/boundsMax as authored) --
// PcgClusterLocalBoundsSSBO (binding 2) is a second, separate, parallel-indexed array carrying
// exactly that local-space AABB, so the two purposes never collide in the same field the way the
// referenced bug once did elsewhere in this codebase.

#include "include/cluster_culling_common.glsl"
#include "include/pcg_instance_draw_common.glsl"

#define COMPRESSED_POOL_SET 0
#define COMPRESSED_POOL_BINDING 1
#include "include/cluster_vertex_decode.glsl"

layout(std430, set = 0, binding = 0) readonly buffer ClusterCullMetadataSSBO {
    ClusterCullMetadata clusters[];
} g_Clusters;

layout(std430, set = 0, binding = 2) readonly buffer PcgClusterLocalBoundsSSBO {
    PcgClusterLocalBounds localBounds[];
} g_LocalBounds;

layout(std430, set = 0, binding = 3) readonly buffer PcgInstanceDataSSBO {
    PcgDrawInstance instances[];
} g_Instances;

// Camera matrices via Push Constants -- same {mat4 view; mat4 proj;} layout as draw.vert/
// ClusterRaster.vert's own CameraPushConstants (exactly 128 bytes, the Vulkan-guaranteed minimum
// push-constant budget -- no room left for anything else in this range, see
// renderer::PcgInstanceDrawPass's own lighting-UBO comment for where sun/ambient parameters live
// instead).
layout(push_constant) uniform PcgCameraPushConstants {
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) flat out uint outMaterialID;

void main() {
    ClusterCullMetadata cluster = g_Clusters.clusters[gl_InstanceIndex];
    PcgClusterLocalBounds lb = g_LocalBounds.localBounds[gl_InstanceIndex];

    // physicalPageIndex is recovered from vertexOffset (== physicalPageIndex * CLUSTER_MAX_VERTICES
    // by construction, see renderer::GeometryDecompressionPass / ClusterCullMetadata::vertexOffset's
    // documented contract, identical to ClusterRaster.vert's own derivation).
    uint physicalPageIndex = cluster.vertexOffset / CLUSTER_MAX_VERTICES;
    uint localVertexIndex = uint(gl_VertexIndex) - cluster.vertexOffset;
    uint pageByteBase = physicalPageIndex * CLUSTER_PAGE_SIZE_BYTES;

    // Decoded directly from the compressed pool using the LOCAL-space bounds -- see this file's own
    // header comment for why this is g_LocalBounds, not cluster.boundsMin/boundsMax.
    vec3 localPos = DecodeClusterPosition(pageByteBase, localVertexIndex, lb.boundsMin, lb.boundsMax);
    vec3 localNormal = DecodeClusterNormal(pageByteBase, localVertexIndex);

    // cluster.entityID is THIS pipeline's own repurposed "PCG instance slot index" -- see this
    // file's own header comment.
    PcgDrawInstance inst = g_Instances.instances[cluster.entityID];

    vec4 worldPos4 = inst.localToWorld * vec4(localPos, 1.0);
    outWorldPos = worldPos4.xyz;
    // Uniform-scale assumption (Phase 0.2 scope, see renderer::PcgInstanceDrawPass's own class
    // comment): a fully correct non-uniform-scale normal transform needs the inverse-transpose of
    // the upper-left 3x3 of localToWorld, which this pipeline does not currently upload a second
    // matrix for (every Phase 0.2 smoke-test instance uses uniform scale, so mat3(localToWorld) is
    // exact for them) -- flag for Phase 4 if non-uniform-scale PCG instances are ever introduced.
    outWorldNormal = normalize(mat3(inst.localToWorld) * localNormal);
    outMaterialID = cluster.materialID;

    gl_Position = camera.proj * camera.view * worldPos4;
}
