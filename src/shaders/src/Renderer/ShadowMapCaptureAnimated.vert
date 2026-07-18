#version 460
#extension GL_GOOGLE_include_directive : enable

// VSM advanced roadmap, Feature 1 (live per-entity transforms): per-entity animated Fallback Mesh
// capture for renderer::VirtualShadowMapPass -- replicates ClusterRaster.vert's own vertex
// transform sequence (local-space spline bend -> rigid rotation/translation -> WPO sway ->
// enhanced displacement) via #include, so a shadow page render always matches THIS frame's actual
// on-screen deformed silhouette (Tube, entity 6, spline; TorusKnot, entity 10, enhanced
// displacement) instead of a frozen rest-pose shadow.
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
// Skeletal-animation feature: this shader does NOT apply ApplySkeletalSkinning (documented, known
// scope limitation -- the procedural creature's shadow is cast from its BIND POSE, not its
// animated pose; see project documentation for this deliberate deviation). Only included for the
// SKELETAL_MAX_DEVIATION constant displacement_bounds.glsl references unconditionally below (this
// file has no bone-matrices SSBO binding of its own).
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
