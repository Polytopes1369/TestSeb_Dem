#version 460
#extension GL_GOOGLE_include_directive : enable

// Include custom structures
#include "include/struct_custo.glsl"

// Bindless Buffers (Storage Buffers)
layout(std430, binding = 0) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer IndexBuffer  { uint indices[]; };
// Per-entity self-rotation (one entry per meshID), re-uploaded from the CPU every frame.
layout(std430, binding = 3) readonly buffer EntityTransformBuffer { EntityTransform entityTransforms[]; };

// Camera matrices via Push Constants
layout(push_constant) uniform CameraPushConstants {
    mat4 view;
    mat4 proj;
} camera;

// Output to Fragment Shader
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorldPos;

void main() {
    // 1. Fetch the index from the Index Buffer
    uint vertexIndex = indices[gl_VertexIndex];

    // 2. Fetch the corresponding Vertex from the Bindless Vertex Buffer
    Vertex v = vertices[vertexIndex];

    // 3. Spin the entity around its own world-space center on all 3 axes: the compute
    // generators already baked each primitive's position around its grid-slot center, so
    // recovering the pre-baked local offset (v.position - center) and re-applying it after
    // rotation keeps the entity pinned in place while it tumbles.
    EntityTransform xform = entityTransforms[v.meshID];
    mat3 rotation = mat3(xform.rotation);
    vec3 localPos = v.position - xform.center;
    vec3 worldPos = xform.center + rotation * localPos;
    vec3 worldNormal = rotation * v.normal;

    // 4. Pass data to Fragment Shader
    outWorldPos = worldPos;
    outNormal = worldNormal;

    // 5. Transform position
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}