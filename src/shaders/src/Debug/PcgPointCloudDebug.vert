#version 460

// Debug-only wireframe box gizmo per PCG point (renderer::debug::PcgPointCloudDebugView, PCG
// roadmap Phase 7.2). Draws one axis-aligned local-space box (pcg::PcgPoint::boundsMin/boundsMax)
// per instance, transformed by that SAME point's own localToWorld matrix (CPU-composed via
// pcg::PcgPoint::GetLocalToWorld() -- see PcgPointData.h's own comment for the exact
// Translate*FromQuat*Scale composition -- NOT re-derived here). VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
// 12 box edges = 24 vertices, entirely generated from gl_VertexIndex/gl_InstanceIndex -- no vertex
// or index buffer of its own, same bindless "SSBO-driven, no VkBuffer vertex input" convention as
// DebugText.vert. This whole pipeline is only ever created/dispatched from Debug-only C++ code
// (#ifndef NDEBUG), so it never ships in the Release .exe (CLAUDE.md's debug-tooling exclusion
// rule) even though this .vert/.frag file pair itself is compiled into shaders/ unconditionally by
// CMake's blanket shader-compile step.

// Byte-for-byte mirror of PcgPointCloudDebugView.h's own GpuPointGizmoInstance struct (std430
// layout): a mat4 (64 bytes, always a whole number of 16-byte column slots) followed by 2x
// (vec3 + trailing scalar packed into the SAME 16-byte std430 slot -- the same idiom every other
// std430 struct in this codebase already uses, e.g. VegetationCommon.glsl's VegetationInstance).
struct PcgPointGizmoInstance {
    mat4 localToWorld;
    vec3 boundsMin; float density; // Local-space AABB min corner + this point's own density [0,1].
    vec3 boundsMax; float _pad0;
};

layout(std430, set = 0, binding = 0) readonly buffer PointInstancesSSBO {
    PcgPointGizmoInstance instances[];
} g_Points;

layout(push_constant) uniform PcgPointCloudPushConstants {
    mat4 viewProj; // proj * view, this frame's camera -- same convention as every other forward pass' own push constant (e.g. VegetationInstanced.vert's RenderParamsUBO.viewProj).
} pc;

layout(location = 0) out vec3 outColor;

// 8 corners of a unit box, indexed 0..7 by an (x,y,z) bit pattern: bit0 = x (0=min,1=max),
// bit1 = y, bit2 = z. The 12 box edges below are expressed as pairs of these corner indices,
// grouped by axis (4 edges running along X, 4 along Y, 4 along Z) -- 24 total vertices for a
// LINE_LIST draw (vertexCount=24, one edge-endpoint per gl_VertexIndex).
const uint kEdgeCornerIndices[24] = uint[24](
    0u, 1u,  2u, 3u,  4u, 5u,  6u, 7u, // X-axis edges
    0u, 2u,  1u, 3u,  4u, 6u,  5u, 7u, // Y-axis edges
    0u, 4u,  1u, 5u,  2u, 6u,  3u, 7u  // Z-axis edges
);

// Resolves one of the 8 unit-box corner indices above into an actual local-space position,
// picking each axis' component from `boundsMin` (bit clear) or `boundsMax` (bit set).
vec3 CornerPosition(uint cornerIndex, vec3 boundsMin, vec3 boundsMax) {
    float x = ((cornerIndex & 1u) != 0u) ? boundsMax.x : boundsMin.x;
    float y = ((cornerIndex & 2u) != 0u) ? boundsMax.y : boundsMin.y;
    float z = ((cornerIndex & 4u) != 0u) ? boundsMax.z : boundsMin.z;
    return vec3(x, y, z);
}

void main() {
    PcgPointGizmoInstance inst = g_Points.instances[gl_InstanceIndex];

    uint cornerIndex = kEdgeCornerIndices[gl_VertexIndex];
    vec3 localPos = CornerPosition(cornerIndex, inst.boundsMin, inst.boundsMax);
    vec3 worldPos = (inst.localToWorld * vec4(localPos, 1.0)).xyz;

    gl_Position = pc.viewProj * vec4(worldPos, 1.0);

    // Density -> color mapping (documented in PcgPointCloudDebugView.h's own header comment): a
    // simple linear lerp from red (density == 0.0) to green (density == 1.0), read unambiguously
    // as "weak/strong" at a glance -- this class' own design decision, not a literal UE5.8 port.
    float d = clamp(inst.density, 0.0, 1.0);
    outColor = mix(vec3(1.0, 0.15, 0.1), vec3(0.15, 1.0, 0.2), d);
}
