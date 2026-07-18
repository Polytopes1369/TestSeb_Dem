#version 460
#extension GL_GOOGLE_include_directive : enable

// Generalized Nanite Tessellation (renderer::TessellationPass): tessellation evaluation shader --
// barycentric-interpolates position/normal from the 3 control points (renderer::
// TessellationPass's own patch, patchControlPoints=3), samples displacement_noise.glsl's fbm
// at the interpolated WORLD-SPACE position (not UV -- see that file's own header comment),
// offsets the position along the interpolated normal, and recomputes the true perturbed normal via
// central differencing -- then owns the final clip-space projection (gl_Position), since it alone
// knows the DISPLACED surface's true position (see Tessellation.vert's own comment on why that
// shader deliberately leaves gl_Position unset). Runs identically for every tessellated entity's
// own draw call.
//
// Crack-free guarantee: two adjacent triangles share exactly the same 2 control points on a common
// edge (the Fallback Mesh is a proper indexed mesh) AND compute the same gl_TessLevelOuter for
// that edge (.tesc), AND SampleDisplacement is a pure function of the interpolated world position
// -- so both triangles' shared-edge tessellated vertices land at IDENTICAL displaced positions,
// keeping the displaced surface seamless.

#include "include/displacement_noise.glsl"

layout(triangles, fractional_odd_spacing, ccw) in;

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

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) in vec3 inWorldNormal[];

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;

void main() {
    vec3 basePos = gl_TessCoord.x * inWorldPos[0] + gl_TessCoord.y * inWorldPos[1] + gl_TessCoord.z * inWorldPos[2];
    vec3 baseNormal = normalize(gl_TessCoord.x * inWorldNormal[0] + gl_TessCoord.y * inWorldNormal[1] + gl_TessCoord.z * inWorldNormal[2]);

    float displacement = SampleDisplacement(basePos, pc.displacementScale);
    vec3 displacedPos = basePos + baseNormal * displacement;
    vec3 displacedNormal = ComputeDisplacedNormal(basePos, baseNormal, pc.displacementScale);

    outWorldPos = displacedPos;
    outWorldNormal = displacedNormal;
    gl_Position = pc.viewProj * vec4(displacedPos, 1.0);
}
