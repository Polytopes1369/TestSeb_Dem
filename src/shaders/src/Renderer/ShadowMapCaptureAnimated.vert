#version 460
#extension GL_GOOGLE_include_directive : enable

// VSM advanced roadmap, Feature 1 (live per-entity transforms): per-entity animated Fallback Mesh
// capture for renderer::VirtualShadowMapPass -- replicates ClusterRaster.vert's own vertex
// transform sequence (local-space skeletal skinning -> spline bend -> rigid rotation/translation ->
// WPO sway -> enhanced displacement) via #include, so a shadow page render always matches THIS
// frame's actual on-screen deformed silhouette (the procedural creature, skeletal skinning; Tube,
// entity 6, spline; TorusKnot, entity 10, enhanced displacement) instead of a frozen rest-pose
// shadow.
//
// renderer::VirtualShadowMapPass::RenderPage() now issues one vkCmdDrawIndexed per entity (see
// EntityDrawRange) instead of one merged draw covering every entity at once, so gl_InstanceIndex
// is always 0 here -- there is no per-cluster ClusterCullMetadataSSBO indirection like
// ClusterRaster.vert's gl_InstanceIndex-driven lookup. `entityID` instead arrives via a push
// constant, one per draw.
//
// Deliberately a SEPARATE file from ShadowMapCapture.vert (left completely untouched -- still used
// by the separate, dead-but-kept renderer::ShadowMapPass, see project memory
// feedback_file_deletion_blocked) and from ClusterRaster.vert (cluster-granular, reads a different
// vertex format from a different buffer entirely). This shader is entity-granular, reading
// geometry::FallbackVertex directly from renderer::VirtualShadowMapPass's own combined vertex
// buffer via ordinary vertex-input attributes (position + uv; the vertex's `normal` field stays
// unbound, unused by a depth-only capture).
//
// maxWPOAmplitude/maskTextureIndex: unlike ClusterRaster.vert (which reads
// ClusterCullMetadata::maxWPOAmplitude/maskTextureIndex, both possibly INFLATED by
// displacement_bounds.glsl's InflateDisplacementBound() at cook time, then un-inflated back via
// GetOriginalWPOAmplitude() immediately before use), the push constant values below arrive ALREADY
// as the entity's TRUE authored values -- renderer::VirtualShadowMapPass::Init() reads them
// straight from geometry::GetEntityMaterialProperties(), the exact same source ClusterDAG.cpp
// itself reads from before any inflation is ever applied. No un-inflation step (and therefore no
// #include of displacement_bounds.glsl) is needed here at all.

#include "include/struct_custo.glsl"
#include "include/wpo_deformation.glsl"
#include "include/enhanced_displacement.glsl"
#include "include/spline_deformation.glsl"
// Skeletal-animation feature (VSM shadow-capture fix): applies ApplySkeletalSkinning, mirroring
// ClusterRaster.vert's own call site exactly (local space, before the spline bend/rigid rotation
// below) -- see skeletal_animation.glsl's own header comment for why ComputeChainSkinWeights
// (rather than a decoded ClusterVertexSkin) is what supplies the bone indices/weights here: this
// shader reads plain geometry::FallbackVertex (position/normal/uv only, no baked skin data) via
// ordinary vertex-input attributes, not the compressed cluster pool ClusterRaster.vert decodes.
#include "include/skeletal_animation.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(std430, set = 0, binding = 0) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(std430, set = 0, binding = 1) readonly buffer EntityDataBuffer {
    EntityData entityData[];
};

// Uploaded once per frame by renderer::ClusterRenderPipeline (vkCmdUpdateBuffer) -- see
// wpo_deformation.glsl's ApplyWPODeformation for how globalTime is used. Byte-identical layout to
// ClusterRaster.vert's own WPOGlobalsUBO (same underlying C++ struct/buffer, just bound at a
// different set/binding here).
layout(std140, set = 0, binding = 2) uniform WPOGlobalsUBO {
    float globalTime;
    float enhancedDisplacementDebugMultiplier;
    float splineDeformationDebugMultiplier;
    float _pad2;
} g_WPOGlobals;

// This scene's one authored Hermite bend curve (renderer::ClusterRenderPipeline's own
// m_SplineControlPointsBuffer, uploaded once at Init) -- see spline_deformation.glsl's own header
// comment for the local-space-before-rotation contract.
layout(std430, set = 0, binding = 3) readonly buffer SplineControlPointsSSBO {
    SplineControlPoint splineControlPoints[SPLINE_CONTROL_POINT_COUNT];
};

// Skeletal-animation feature (VSM shadow-capture fix): this frame's bone-matrices SSBO
// (animation::SkeletalAnimator::GetBoneMatricesBuffer(), uploaded once per frame by
// SkeletalAnimator::RecordUpdate) -- bound read-only at binding 5, the first free slot past this
// pass's own bindings 0-3 (binding 4 is the fragment-only bindless mask array, see
// renderer::VirtualShadowMapPass::Init's own layout comment). See skeletal_animation.glsl's own
// header comment for the byte layout and ApplySkeletalSkinning's own header comment for the
// local-space-before-rotation contract.
layout(std430, set = 0, binding = 5) readonly buffer SkeletalBoneMatricesSSBO {
    mat4 boneMatrices[SKELETAL_MAX_BONES];
};

// entityID indexes EntityDataBuffer/EntityTransformBuffer (one draw == one entity, see this file's
// own header comment). maxWPOAmplitude/maskTextureIndex are this entity's precomputed, TRUE
// (non-inflated) values -- see this file's own header comment.
layout(push_constant) uniform ShadowCaptureConstants {
    mat4 lightViewProj;
    uint entityID;
    float maxWPOAmplitude;
    uint maskTextureIndex;
} pc;

// Interpolated UV + flat mask index -- consumed only by ShadowMapCaptureMasked.frag's opacity-mask
// discard (Feature 3, mask_sampling.glsl); both are exact no-ops for the unmasked pipeline, which
// has no fragment stage at all and therefore never reads these outputs.
layout(location = 0) out vec2 outUV;
layout(location = 1) flat out uint outMaskTextureIndex;

void main() {
    EntityData ed = entityData[pc.entityID];
    EntityTransform xform = entityTransforms[ed.meshID];
    mat3 rotation = mat3(xform.rotation);

    // inPosition is the Fallback Mesh's baked (rest-pose) position -- the same space
    // ClusterRaster.vert calls "worldPos" immediately after DecodeClusterPosition(), before
    // subtracting EntityTransform.center.
    vec3 worldPos = inPosition;
    vec3 localPos = worldPos - xform.center;

    // Skeletal-animation feature (VSM shadow-capture fix): linear-blend vertex skinning, applied in
    // LOCAL space BEFORE the per-entity rigid rotation (and before spline bend below) -- mirrors
    // ClusterRaster.vert's own call site exactly (same ordering rationale: skinning is an intrinsic
    // rest-pose mesh-shape property, not a world-space effect). Bone indices/weights are derived
    // analytically from this vertex's own local-space X coordinate (ComputeChainSkinWeights,
    // skeletal_animation.glsl) rather than decoded from a baked ClusterVertexSkin buffer -- see that
    // function's own header comment for why this Fallback Mesh vertex has no such buffer to decode.
    if (GetFlag(ed.flags, ENTITY_FLAG_IS_SKELETALLY_ANIMATED)) {
        uvec4 boneIndices;
        vec4 boneWeights;
        ComputeChainSkinWeights(localPos.x, boneIndices, boneWeights);
        localPos = ApplySkeletalSkinning(localPos, boneIndices, boneWeights, boneMatrices);
    }

    // Spline bend, in LOCAL space, BEFORE the rigid rotation -- see spline_deformation.glsl's own
    // header comment, mirrored verbatim from ClusterRaster.vert.
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_SPLINE_DEFORMATION)) {
        localPos = mix(localPos, ApplySplineDeformation(localPos, splineControlPoints), g_WPOGlobals.splineDeformationDebugMultiplier);
    }

    worldPos = xform.translation + xform.center + rotation * localPos;

    // WPO sway: pc.maxWPOAmplitude is this entity's TRUE authored amplitude (see this file's own
    // header comment) -- no GetOriginalWPOAmplitude() un-inflation needed, unlike ClusterRaster
    // .vert. pc.entityID stands in for ClusterRaster.vert's per-cluster phase seed
    // (cluster.clusterID) -- coarser (one sway phase per entity instead of per cluster), an
    // accepted simplification since the Fallback Mesh has no cluster granularity to begin with.
    worldPos = ApplyWPODeformation(worldPos, pc.entityID, pc.maxWPOAmplitude, g_WPOGlobals.globalTime);

    // Multi-octave enhanced displacement, applied ADDITIVELY right after the WPO sway -- see
    // enhanced_displacement.glsl's own header comment, mirrored verbatim from ClusterRaster.vert
    // (same pc.entityID-as-noise-seed substitution as the WPO sway call above).
    if (GetFlag(ed.flags, ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT)) {
        vec3 displaced = ApplyEnhancedDisplacement(worldPos, xform.center, pc.entityID, g_WPOGlobals.globalTime);
        worldPos = worldPos + (displaced - worldPos) * g_WPOGlobals.enhancedDisplacementDebugMultiplier;
    }

    outUV = inUV;
    outMaskTextureIndex = pc.maskTextureIndex;

    gl_Position = pc.lightViewProj * vec4(worldPos, 1.0);
}
