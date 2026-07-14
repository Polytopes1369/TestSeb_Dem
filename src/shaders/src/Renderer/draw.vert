#version 460
#extension GL_GOOGLE_include_directive : enable

// Include custom structures
#include "include/struct_custo.glsl"

// Bindless Buffers (Storage Buffers)
layout(std430, binding = 0) readonly buffer VertexBuffer { Vertex vertices[]; };
layout(std430, binding = 1) readonly buffer IndexBuffer  { uint indices[]; };

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

    // 3. Pass data to Fragment Shader
    outWorldPos = v.position;
    outNormal = v.normal;
    
    // 4. Transform position
    gl_Position = camera.proj * camera.view * vec4(v.position, 1.0);
}