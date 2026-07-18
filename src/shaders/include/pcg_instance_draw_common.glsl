#ifndef PCG_INSTANCE_DRAW_COMMON_GLSL
#define PCG_INSTANCE_DRAW_COMMON_GLSL

// Phase 0.2 (UE5.8-parity PCG roadmap, "PCG Instance Draw Path"): GLSL-side mirrors of
// renderer::PcgInstanceDrawPass's two private, pipeline-local SSBO element types (see that class'
// own header comment for the full design rationale). Both are consumed exclusively by
// PcgInstanceDraw.vert -- unlike ClusterCullMetadata (cluster_culling_common.glsl), which is shared
// with every other Nanite cluster consumer, these two structs are private to this one draw path
// and therefore free to define their own field layout without needing to match any on-disk format.

// Byte-for-byte, std430-compatible mirror of renderer::PcgInstanceDrawPass::GpuInstanceData -- one
// entry per acquired PCG instance slot (NOT one per candidate cluster; many clusters can share the
// same instance slot index, exactly like geometry::ClusterIndexEntry::entityID lets many clusters
// share one scene entity). `localToWorld` is composed CPU-side (maths::mat4::Translate(position) *
// maths::mat4::FromQuat(rotation) * maths::mat4::Scale(scale), same TRS convention as
// pcg::PcgPoint::GetLocalToWorld) so this shader never needs to reconstruct a quaternion rotation
// itself. `materialID` indexes MaterialParamsSSBO (material_params.glsl) exactly like
// ClusterCullMetadata::materialID already does for the shared opaque Nanite path.
struct PcgDrawInstance {
    mat4 localToWorld;
    uint materialID;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

// Byte-for-byte, std430-compatible mirror of renderer::PcgInstanceDrawPass::LocalBoundsGpu -- the
// LOCAL-space (i.e. as-authored, NOT transformed by any instance's world transform) AABB
// geometry::ClusterIndexEntry::boundsMin/boundsMax originally held for this candidate cluster.
//
// IMPORTANT: this is a SEPARATE array from ClusterCullMetadata's own boundsMin/boundsMax, which
// this draw path instead fills with the WORLD-space AABB (required for renderer::ClusterCullingPass's
// frustum test to be correct once a per-instance world transform is involved -- see
// PcgInstanceDraw.vert's own header comment and project memory
// project_cluster_bounds_decode_vs_culling.md for the exact gotcha this avoids: boundsMin/boundsMax
// also feeds vertex dequantization, so overwriting ONE shared field with a rotated/translated copy
// would silently corrupt every vertex position decoded from it).
struct PcgClusterLocalBounds {
    vec3 boundsMin;
    float _pad0;
    vec3 boundsMax;
    float _pad1;
};

#endif // PCG_INSTANCE_DRAW_COMMON_GLSL
