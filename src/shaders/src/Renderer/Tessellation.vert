#version 460
#extension GL_GOOGLE_include_directive : enable

// Generalized Nanite Tessellation (renderer::TessellationPass, generalized from the earlier
// single-hardcoded-entity "HeroTessellationPass" -- see that class' own class comment): forward-
// rendered, screen-space-adaptively-tessellated, procedurally-displaced ENTITY vertex shader --
// runs once per tessellated entity's own draw call (renderer::TessellationPass::RecordDraw loops
// one vkCmdDrawIndexed per core::EntityFlags::IsTessellated entity, rebinding only push constants
// between them). Every such entity's clusters are excluded from the Nanite VisBuffer path entirely
// via core::EntityFlags::IsTransparent, see VulkanContext::BuildEntityData()'s own comment, and
// rendered here instead against its own Fallback Mesh draw range, reinterpreted as 3-control-point
// triangle patches). Plain vertex-attribute input (geometry::FallbackVertex), same convention as
// TransparentForward.vert -- the Fallback Mesh is free-form, full-precision, read once at startup,
// not the bindless compressed-cluster-pool path the opaque raster shaders use.
//
// Unlike a normal vertex shader, this one does NOT compute gl_Position: the tessellation
// evaluation stage owns the final clip-space position, since it projects the DISPLACED surface,
// not this (undisplaced) control point. Applies the entity's current rotation to the raw Fallback
// Mesh position/normal -- the exact same "worldPos = center + rotation*(worldPos-center)" formula
// TransparentForward.vert's own inline transform uses (this codebase duplicates this formula per
// shader rather than sharing it, see that shader's own comment); a no-op for every tessellated
// entity this phase assigns today (config::ENTITY_SELF_ROTATION_ENABLED off by default), kept
// general so a future rotating tessellated entity would already work. Indexes entityTransforms[]
// directly by pc.entityID (not via an EntityData->meshID indirection like TransparentForward.vert)
// since every entity this pass draws has meshID == entityID (VulkanContext::BuildEntityData()
// assigns dense sequential IDs, see that function's own comment) -- pc.entityID is set fresh per
// draw call by renderer::TessellationPass::RecordDraw's own per-entity loop.

#include "include/struct_custo.glsl"

layout(std430, set = 0, binding = 8) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV; // Unused -- displacement samples world position, not UV (see displacement_noise.glsl's own header comment).

// Byte-for-byte mirror of TessellationConstants in .tesc/.tese/.frag -- flat scalar fields
// throughout, matching this codebase's established push-constant convention (see
// GlobalSDFCompositePC's own comment). `materialID` selects this draw's own entity's slot in the
// full MaterialParameters[kMaxMaterials] SSBO (binding 7, .frag-only) -- unused by this stage, but
// declared here too since every stage sharing one push_constant range must agree on its full byte
// layout (this codebase's own established convention, see .frag's own identical declaration).
layout(push_constant) uniform TessellationConstants {
    mat4 viewProj;
    float cameraPositionWorldX, cameraPositionWorldY, cameraPositionWorldZ;
    float _pad0;
    uint entityID;
    uint traceMode;
    uint frameIndex;
    uint entityCount;
    float viewportWidth, viewportHeight;
    float displacementScale;
    uint materialID;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;

void main() {
    EntityTransform xform = entityTransforms[pc.entityID];
    mat3 rotation = mat3(xform.rotation);
    outWorldPos = xform.translation + xform.center + rotation * (inPosition - xform.center);
    outWorldNormal = rotation * inNormal;
}
