#version 460
#extension GL_GOOGLE_include_directive : enable

// Vegetation scatter (UE5.8 rendering-parity gap G2), instanced draw vertex stage. NO bound vertex
// attribute buffer -- the base archetype mesh's vertices are fetched from an SSBO via gl_VertexIndex
// (the index buffer bound for the indexed indirect draw resolves each triangle's global vertex
// index), exactly the "read the vertex SSBO in-shader" idiom ClusterRaster.vert already uses. Each
// instance's transform is fetched via gl_InstanceIndex, indexed through this archetype's compacted
// visible-instance segment (renderer::VegetationInstanceCull.comp writes it) -- so this shader only
// ever transforms real, frustum/HZB-surviving instances.

#include "include/struct_custo.glsl"        // Vertex
#include "include/VegetationCommon.glsl"     // VegetationInstance, VegetationTint

layout(std430, set = 0, binding = 0) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(std430, set = 0, binding = 1) readonly buffer InstanceBuffer { VegetationInstance instances[]; };
layout(std430, set = 0, binding = 2) readonly buffer VisibleIndexBuffer { uint visibleIndices[]; };

layout(std140, set = 1, binding = 0) uniform RenderParamsUBO {
    mat4 viewProj;
    vec3 cameraPos; float _pad0;
    vec3 sunDirection; float sunIntensity; // Fragment-stage fields, declared here too so the std140 block matches byte-for-byte across stages.
    vec3 sunColor; float _pad1;
} g_Params;

layout(push_constant) uniform PushConstants {
    uint archetypeSegmentBase; // = archetype * maxInstances -- base of this archetype's segment in visibleIndices[].
    uint archetype;            // kVegArchetype* -- selects the fragment base color.
    uint _pad0;
    uint _pad1;
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 outTint;

void main() {
    uint visIdx = visibleIndices[pc.archetypeSegmentBase + uint(gl_InstanceIndex)];
    VegetationInstance inst = instances[visIdx];
    Vertex v = vertices[gl_VertexIndex];

    // Yaw rotation about world +Y (the only per-instance rotation a scatter needs) + uniform scale.
    float cy = cos(inst.yaw);
    float sy = sin(inst.yaw);
    vec3 local = v.position * inst.scale;
    vec3 rotated = vec3(cy * local.x + sy * local.z, local.y, -sy * local.x + cy * local.z);
    vec3 worldPos = inst.position + rotated;

    vec3 n = v.normal;
    vec3 rotatedNormal = vec3(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z);

    outWorldPos = worldPos;
    outWorldNormal = normalize(rotatedNormal);
    outUV = v.uv;
    outTint = VegetationTint(inst.tintSeed, pc.archetype);

    gl_Position = g_Params.viewProj * vec4(worldPos, 1.0);
}
