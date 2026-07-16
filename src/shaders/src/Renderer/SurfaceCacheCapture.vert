#version 460
#extension GL_GOOGLE_include_directive : enable

// Surface Cache capture vertex shader (see renderer::SurfaceCachePass): transforms one Fallback
// Mesh vertex through this Card's orthographic view-projection (renderer::SurfaceCachePass::
// BuildCardViewProj computes one such matrix per card -- LookAt toward the entity's AABB center
// from just outside the card's face, composed with mat4::OrthoVulkan sized to the face's exact
// world-space footprint) and passes world-space position/normal through for
// SurfaceCacheCapture.frag's procedural shading + normal encoding. Plain vertex-attribute input
// (NOT the bindless compressed-cluster-pool path every other raster shader in this codebase uses)
// since the Fallback Mesh is free-form, full-precision, and read once at startup -- see
// FallbackMeshBuilder.h / ClusterFormat.h's FallbackVertex.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV; // Unused here -- no texture/material-binding system yet (see procedural_material.glsl).

layout(push_constant) uniform SurfaceCaptureConstants {
    mat4 viewProj;
    uint entityID;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;

void main() {
    outWorldPos = inPosition;
    outWorldNormal = inNormal;
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
}
