#ifndef CLUSTER_CULLING_COMMON_GLSL
#define CLUSTER_CULLING_COMMON_GLSL

// Shared data layout for the GPU-driven cluster visibility pass (see
// src/shaders/src/Culling/ClusterFrustumCull.comp and renderer::ClusterCullingPass). Two structs
// are declared here, both consumed only by that one shader today, factored out into their own
// include so a future second cull pass (e.g. an HZB occlusion pass sitting alongside this one,
// see HZBPass.h) can reuse the exact same metadata/view-params layout without duplicating it.

// GLSL-side, std430-compatible mirror of geometry::ClusterIndexEntry (see ClusterFormat.h). The
// on-disk struct is `#pragma pack(1)` with 8-bit cone fields and cannot be bound directly as an
// SSBO -- renderer::ClusterCullingPass::ClusterCullMetadata is the exact C++ counterpart this must
// match field-for-field (byte offset for byte offset, verified by that struct's static_assert);
// the CPU-side upload step is responsible for widening ClusterIndexEntry's int8_t cone fields back
// into normalized floats (dividing by 127, the exact inverse of
// geometry::VirtualGeometryCacheTest.cpp's BuildIndexEntry quantization) before it ever reaches
// this buffer.
struct ClusterCullMetadata {
    vec3 boundsMin;        // Local/world-space AABB min, used by the frustum test.
    float _padBoundsMin;
    vec3 boundsMax;        // Local/world-space AABB max, used by the frustum test.
    float _padBoundsMax;
    vec3 sphereCenter;     // Bounding sphere, used by the backface cone test.
    float sphereRadius;
    vec3 coneAxis;         // Unit vector: every triangle normal in the cluster satisfies dot(n, coneAxis) >= coneCutoff.
    float coneCutoff;      // cos(maxAngleFromAxis), in [-1, 1].
    uint indexCount;        // geometry::ClusterIndexEntry::indexCount -- becomes DrawIndexedIndirectCommand::indexCount.
    uint firstIndex;         // Base offset into the global decompressed index buffer for this cluster.
    uint vertexOffset;       // Base vertex added to every local index before indexing the global vertex buffer.
    uint clusterID;
    float maxWPOAmplitude;  // Worst-case world-space vertex displacement the WPO sway function can apply (see wpo_deformation.glsl).
    uint maskTextureIndex;  // Index into the bindless cutout mask array (mask_sampling.glsl), or 0xFFFFFFFF for "fully opaque".
    // No manual padding needed here (unlike the C++ mirror): the GLSL compiler infers the std430
    // array stride (96 bytes, rounded up from this struct's 16-byte base alignment) on its own.
};

// GLSL-side, std140-compatible mirror of renderer::ClusterCullViewParams: the camera's 6-plane
// frustum plus its world-space position, uploaded once per frame via vkCmdUpdateBuffer into the
// UBO this struct describes (see renderer::ClusterCullingPass::RecordCull).
struct CullingViewParams {
    // Camera frustum, one plane per side (xyz = outward unit normal, w = signed distance from the
    // origin), extracted CPU-side from the combined view-projection matrix (Gribb-Hartmann
    // method, see renderer::ExtractFrustumPlanes). A point p is outside plane i when
    // dot(frustumPlanes[i].xyz, p) + frustumPlanes[i].w < 0.
    vec4 frustumPlanes[6];
    vec3 cameraPositionWorld; // Used as the eye point for the backface cone test.
    float _padCameraPosition;
};

#endif // CLUSTER_CULLING_COMMON_GLSL
