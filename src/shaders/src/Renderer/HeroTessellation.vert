#version 460
#extension GL_GOOGLE_include_directive : enable

// Phase 7a (UE5.8 parity roadmap, hero asset tessellation): forward-rendered, screen-space-
// adaptively-tessellated, procedurally-displaced hero entity vertex shader (see
// renderer::HeroTessellationPass's own class comment for the full rationale -- the hero entity's
// clusters are excluded from the Nanite VisBuffer path entirely via core::EntityFlags::
// IsTransparent, see VulkanContext::BuildEntityData()'s own comment, and rendered here instead
// against its Fallback Mesh, reinterpreted as 3-control-point triangle patches). Plain
// vertex-attribute input (geometry::FallbackVertex), same convention as TransparentForward.vert --
// the Fallback Mesh is free-form, full-precision, read once at startup, not the bindless
// compressed-cluster-pool path the opaque raster shaders use.
//
// Unlike a normal vertex shader, this one does NOT compute gl_Position: the tessellation
// evaluation stage owns the final clip-space position, since it projects the DISPLACED surface,
// not this (undisplaced) control point. Applies the entity's current rotation to the raw Fallback
// Mesh position/normal -- the exact same "worldPos = center + rotation*(worldPos-center)" formula
// TransparentForward.vert's own inline transform uses (this codebase duplicates this formula per
// shader rather than sharing it, see that shader's own comment); a no-op for the static hero
// entity this phase assigns (config::ENTITY_SELF_ROTATION_ENABLED off by default), kept general
// so a future rotating hero entity would already work. Indexes entityTransforms[] directly by
// pc.entityID (not via an EntityData->meshID indirection like TransparentForward.vert) since this
// pass only ever draws ONE fixed entity whose meshID == entityID (VulkanContext::BuildEntityData()
// assigns dense sequential IDs, see that function's own comment).

#include "include/struct_custo.glsl"

layout(std430, set = 0, binding = 8) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV; // Unused -- displacement samples world position, not UV (see displacement_noise.glsl's own header comment).

// Byte-for-byte mirror of HeroTessellationConstants in .tesc/.tese/.frag -- flat scalar fields
// throughout, matching this codebase's established push-constant convention (see
// GlobalSDFCompositePC's own comment).
layout(push_constant) uniform HeroTessellationConstants {
    mat4 viewProj;
    float cameraPositionWorldX, cameraPositionWorldY, cameraPositionWorldZ;
    float _pad0;
    uint entityID;
    uint traceMode;
    uint frameIndex;
    uint entityCount;
    float viewportWidth, viewportHeight;
    float displacementScale;
    float _pad1;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;

void main() {
    EntityTransform xform = entityTransforms[pc.entityID];
    mat3 rotation = mat3(xform.rotation);
    outWorldPos = xform.center + rotation * (inPosition - xform.center);
    outWorldNormal = rotation * inNormal;
}
