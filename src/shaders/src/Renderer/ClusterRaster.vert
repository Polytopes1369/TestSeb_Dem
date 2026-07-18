#version 460
#extension GL_GOOGLE_include_directive : enable

// Hardware rasterization vertex shader for "large" clusters (Nanite-style dual rasterization path
// -- see renderer::ClusterHardwareRasterPass). Dispatched via vkCmdDrawIndexedIndirectCount, one
// VkDrawIndexedIndirectCommand per surviving cluster (produced entirely on the GPU by
// ClusterFrustumCull.comp / ClusterHZBOcclusionCull.comp): every render argument (which cluster,
// how many indices, at what buffer offsets) already comes from that indirect command, so this
// shader is only responsible for turning (gl_VertexIndex, gl_InstanceIndex) into a world-space
// clip-space position, decoded straight from the *compressed* physical page pool.
//
// gl_InstanceIndex identifies the cluster: every indirect command's instanceCount is 1 and
// firstInstance carries that cluster's own index into g_Clusters.clusters[] (see
// ClusterFrustumCull.comp / ClusterHZBOcclusionCull.comp's EmitEarlyDraw/EmitLateDraw), so
// gl_InstanceIndex == firstInstance exactly and re-indexes the very same
// ClusterCullMetadataSSBO the culling pass already read.
//
// gl_VertexIndex is the fixed-function-resolved absolute vertex slot: the index buffer bound via
// vkCmdBindIndexBuffer (renderer::GeometryDecompressionPass::GetDecompressedIndexPoolBuffer(),
// VK_INDEX_TYPE_UINT32, expanded ahead of time by DecompressClusterIndices.comp from
// ClusterData::indices) stores each cluster's *local* index values (0, kMaxClusterVertices)); the
// fixed-function stage adds VkDrawIndexedIndirectCommand::vertexOffset (== this cluster's
// ClusterCullMetadata::vertexOffset, physicalPageIndex * CLUSTER_MAX_VERTICES) before this shader
// ever runs. Subtracting that same vertexOffset back off here recovers the local vertex index
// needed to address the compressed page's SoA vertex block.

#include "include/cluster_culling_common.glsl"
#include "include/struct_custo.glsl"

#define COMPRESSED_POOL_SET 0
#define COMPRESSED_POOL_BINDING 1
#include "include/cluster_vertex_decode.glsl"
#include "include/wpo_deformation.glsl"
#include "include/enhanced_displacement.glsl"
#include "include/spline_deformation.glsl"
#include "include/skeletal_animation.glsl"
#include "include/displacement_bounds.glsl"

layout(std430, set = 0, binding = 0) readonly buffer ClusterCullMetadataSSBO {
    ClusterCullMetadata clusters[];
} g_Clusters;

layout(std430, set = 0, binding = 4) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(std430, set = 0, binding = 5) readonly buffer EntityDataBuffer {
    EntityData entityData[];
};

// Uploaded once per frame by renderer::ClusterRenderPipeline (vkCmdUpdateBuffer) -- see
// wpo_deformation.glsl's ApplyWPODeformation for how globalTime is used. _pad exists only to mirror
// the owning C++ WPOGlobalsUBO struct's explicit 16-byte size; std140 does not require it here.
layout(std140, set = 0, binding = 2) uniform WPOGlobalsUBO {
    float globalTime;
    // Phase 1 (Nanite advanced) debug toggles -- see ClusterRenderPipeline::WPOGlobalsUBO's own
    // comment: 1.0 = full effect, 0.0 = fully off (Release always uploads 1.0, no toggle exists).
    float enhancedDisplacementDebugMultiplier;
    float splineDeformationDebugMultiplier;
    float _pad2;
} g_WPOGlobals;

// Phase 1 (Nanite advanced): this entity's authored Hermite bend curve (renderer::
// ClusterRenderPipeline's m_SplineControlPointsBuffer, uploaded once at Init) -- see
// spline_deformation.glsl's own header comment for the local-space-before-rotation contract.
layout(std430, set = 0, binding = 6) readonly buffer SplineControlPointsSSBO {
    SplineControlPoint splineControlPoints[SPLINE_CONTROL_POINT_COUNT];
};

// Skeletal-animation feature: this frame's bone-matrices SSBO (animation::SkeletalAnimator::
// GetBoneMatricesBuffer(), uploaded once per frame by SkeletalAnimator::RecordUpdate) -- bound
// read-only at binding 7, the first free slot past bindings 0-6 above. See
// skeletal_animation.glsl's own header comment for the byte layout and
// ApplySkeletalSkinning's own header comment for the local-space-before-rotation contract.
layout(std430, set = 0, binding = 7) readonly buffer SkeletalBoneMatricesSSBO {
    mat4 boneMatrices[SKELETAL_MAX_BONES];
};

// Camera matrices via Push Constants -- same layout as draw.vert's CameraPushConstants.
layout(push_constant) uniform CameraPushConstants {
    mat4 view;
    mat4 proj;
} camera;

// Visibility Buffer ClusterID, passed through to ClusterRaster.frag. This is gl_InstanceIndex
// (this cluster's own slot in ClusterCullMetadataSSBO), NOT geometry::ClusterIndexEntry::clusterID
// (the unrelated, persistent, potentially-sparse on-disk cache identifier) -- a later resolve pass
// (renderer::ClusterResolvePass) needs to re-index g_Clusters.clusters[] directly from whatever is
// stored in the VisBuffer to reconstruct per-pixel vertex data, which only a dense array slot
// index supports; the true clusterID remains reachable, if ever needed, via one extra indirection
// (g_Clusters.clusters[slotIndex].clusterID). Local TriangleID is NOT computed here --
// ClusterRaster.frag reads it directly from gl_PrimitiveID, which the fixed-function primitive
// assembly stage resets to 0 at the start of each indirect sub-draw (i.e. each cluster's own
// VkDrawIndexedIndirectCommand), making it exactly this cluster's local triangle ordinal with no
// manual bookkeeping needed here.
layout(location = 0) flat out uint outClusterID;
// Interpolated UV + flat mask index, consumed only by ClusterRaster.frag's opacity-mask discard
// (mask_sampling.glsl) -- both are exact no-ops for any cluster whose maskTextureIndex is the
// kInvalidMaskTextureIndex sentinel (0xFFFFFFFF), the common (non-foliage) case.
layout(location = 1) out vec2 outUV;
layout(location = 2) flat out uint outMaskTextureIndex;

void main() {
    ClusterCullMetadata cluster = g_Clusters.clusters[gl_InstanceIndex];

    // physicalPageIndex is recovered from vertexOffset (== physicalPageIndex * CLUSTER_MAX_VERTICES
    // by construction, see renderer::GeometryDecompressionPass / ClusterCullMetadata::vertexOffset's
    // documented contract) rather than needing a second metadata field -- exact integer division
    // since vertexOffset is always page-aligned.
    uint physicalPageIndex = cluster.vertexOffset / CLUSTER_MAX_VERTICES;
    uint localVertexIndex = uint(gl_VertexIndex) - cluster.vertexOffset;
    uint pageByteBase = physicalPageIndex * CLUSTER_PAGE_SIZE_BYTES;

    // Decoded directly from the compressed pool -- see DecodeClusterPosition
    // (cluster_vertex_decode.glsl) -- no dependency on renderer::GeometryDecompressionPass's
    // vertex pool having already decompressed this exact page this frame.
    vec3 worldPos = DecodeClusterPosition(pageByteBase, localVertexIndex, cluster.boundsMin, cluster.boundsMax);

    // Apply entity self-rotation
    EntityData ed = entityData[cluster.entityID];
    EntityTransform xform = entityTransforms[ed.meshID];
    mat3 rotation = mat3(xform.rotation);
    vec3 localPos = worldPos - xform.center;

    // Skeletal-animation feature: linear-blend vertex skinning, applied in LOCAL space BEFORE the
    // per-entity rigid rotation (and before spline bend below) -- see skeletal_animation.glsl's own
    // header comment for why this ordering matches ApplySplineDeformation's own contract exactly
    // (both are intrinsic rest-pose mesh-shape properties, not world-space effects). Bone
    // indices/weights are decoded straight from the compressed pool, exactly like
    // position/normal/UV -- see cluster_vertex_decode.glsl's own DecodeClusterSkin.
    if (GetFlag(ed.flags, ENTITY_FLAG_IS_SKELETALLY_ANIMATED)) {
        uvec4 boneIndices;
        vec4 boneWeights;
        DecodeClusterSkin(pageByteBase, localVertexIndex, boneIndices, boneWeights);
        localPos = ApplySkeletalSkinning(localPos, boneIndices, boneWeights, boneMatrices);
    }

    // Phase 1 (Nanite advanced): spline bend, applied in LOCAL space BEFORE the per-entity rigid
    // rotation -- see spline_deformation.glsl's own header comment for why this ordering is
    // required (a spline bend is an intrinsic mesh-shape property, not a world-space effect).
    // Mixed toward the undeformed local position by the debug multiplier rather than scaled
    // directly (scaling a position would collapse it toward the origin, not toward "undeformed").
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_SPLINE_DEFORMATION)) {
        localPos = mix(localPos, ApplySplineDeformation(localPos, splineControlPoints), g_WPOGlobals.splineDeformationDebugMultiplier);
    }

    worldPos = xform.translation + xform.center + rotation * localPos;

    worldPos = ApplyWPODeformation(worldPos, cluster.clusterID, GetOriginalWPOAmplitude(cluster.maxWPOAmplitude, ed.flags), g_WPOGlobals.globalTime);

    // Phase 1 (Nanite advanced): multi-octave enhanced displacement, applied ADDITIVELY right after
    // the existing WPO sway -- see enhanced_displacement.glsl's own header comment. The debug
    // multiplier scales the DELTA (not the absolute position) toward zero, since
    // ApplyEnhancedDisplacement returns an absolute position, not an offset.
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT)) {
        vec3 displaced = ApplyEnhancedDisplacement(worldPos, xform.center, cluster.clusterID, g_WPOGlobals.globalTime);
        worldPos = worldPos + (displaced - worldPos) * g_WPOGlobals.enhancedDisplacementDebugMultiplier;
    }

    outClusterID = uint(gl_InstanceIndex);
    outUV = DecodeClusterUV(pageByteBase, localVertexIndex);
    outMaskTextureIndex = cluster.maskTextureIndex;

    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}
