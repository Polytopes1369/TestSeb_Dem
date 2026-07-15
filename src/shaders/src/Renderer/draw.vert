#version 460
#extension GL_GOOGLE_include_directive : enable

// Include custom structures
#include "include/struct_custo.glsl"

// Bindless Buffers (Storage Buffers)
layout(std430, binding = 0) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer IndexBuffer  { uint indices[]; };
// Per-entity self-rotation (one entry per meshID), re-uploaded from the CPU every frame.
layout(std430, binding = 3) readonly buffer EntityTransformBuffer { EntityTransform entityTransforms[]; };
// CPU-authored entity records (meshID assigned via core::IDManager, see BuildEntityData() /
// UploadEntityData() in VulkanContext.cpp), uploaded once at startup.
layout(std430, binding = 4) readonly buffer EntityDataBuffer { EntityData entityData[]; };

// Camera matrices via Push Constants
layout(push_constant) uniform CameraPushConstants {
    mat4 view;
    mat4 proj;
} camera;

// Output to Fragment Shader
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorldPos;
// meshID must be "flat" (no interpolation) since it is an integer identifier,
// not an interpolable quantity. It lets the fragment shader derive a
// per-primitive procedural color without needing an extra CPU-uploaded buffer.
layout(location = 2) flat out uint outMeshID;

void main() {
    // 1. Fetch the index from the Index Buffer
    uint vertexIndex = indices[gl_VertexIndex];

    // 2. Fetch the corresponding Vertex from the Bindless Vertex Buffer
    Vertex v = vertices[vertexIndex];

    // 3. Look up this vertex's entity record (CPU-authored, meshID assigned by IDManager),
    // then spin it around its own world-space center on all 3 axes: the compute generators
    // already baked each primitive's position around its grid-slot center, so recovering the
    // pre-baked local offset (v.position - center) and re-applying it after rotation keeps
    // the entity pinned in place while it tumbles.
    EntityData ed = entityData[v.meshID];
    EntityTransform xform = entityTransforms[ed.meshID];
    mat3 rotation = mat3(xform.rotation);
    vec3 localPos = v.position - xform.center;
    vec3 worldPos = xform.center + rotation * localPos;
    vec3 worldNormal = rotation * v.normal;

    // 4. Pass data to Fragment Shader
    outWorldPos = worldPos;
    outNormal = worldNormal;
    outMeshID = v.meshID;

    // 5. Transform position
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}