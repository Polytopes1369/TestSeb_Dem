#version 460
#extension GL_GOOGLE_include_directive : enable

// Phase 7c (UE5.8 parity roadmap, water/erosion): forward-rendered water plane vertex shader --
// near-literal copy of HeroTessellation.vert's own entity-transform pattern (see
// renderer::WaterForwardPass's own class comment for why this is a sibling pass, not a
// generalization of TransparentForwardPass). The water entity is statically flat (no vertex
// displacement -- wave perturbation is shading-only, applied to the NORMAL in WaterForward.frag,
// see that file's own header comment), so this shader is a pure passthrough beyond the entity
// transform (an exact no-op for this static entity today, kept general for consistency with every
// other entity-transform-aware vertex shader in this codebase).

#include "include/struct_custo.glsl"

layout(std430, set = 0, binding = 1) readonly buffer EntityTransformBuffer {
    EntityTransform entityTransforms[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV; // Unused -- no texture/material-binding system yet.

layout(push_constant) uniform WaterForwardConstants {
    mat4 viewProj;
    float cameraPositionWorldX, cameraPositionWorldY, cameraPositionWorldZ;
    float _pad0;
    uint entityID;
    uint traceMode;
    uint frameIndex;
    uint entityCount;
    float viewportWidth, viewportHeight;
    float timeSeconds;
    float _pad1;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;

void main() {
    EntityTransform xform = entityTransforms[pc.entityID];
    mat3 rotation = mat3(xform.rotation);
    vec3 worldPos = xform.center + rotation * (inPosition - xform.center);
    outWorldPos = worldPos;
    outWorldNormal = rotation * inNormal;
    gl_Position = pc.viewProj * vec4(worldPos, 1.0);
}
