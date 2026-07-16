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

// Output to Fragment Shader -- Visibility Buffer IDs only (no normal/world-position/lighting
// data: draw.frag no longer shades anything, it only rasterizes these two identifiers). Both
// must be "flat" (no interpolation) since they are integer identifiers, not interpolable
// quantities -- every vertex of a triangle carries the exact same triangleID (see below) and,
// today, the same clusterID (meshID stand-in), so flat-shading from any one of the triangle's
// vertices (the provoking vertex) is exact, not an approximation.
layout(location = 0) flat out uint outClusterID;
layout(location = 1) flat out uint outTriangleID;

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

    // 4. Visibility Buffer IDs.
    // ClusterID: this draw call still renders the flat, non-clustered per-entity SSBOs built by
    // VulkanContext::GenerateGeometry() (see the geometry-clustering pipeline under
    // src/geometry/ClusterFormat.h / src/renderer/Cluster*CullingPass.h for the real, not-yet-wired
    // clustered path) -- so meshID stands in for ClusterID today. Once draws are issued per
    // geometry::ClusterIndexEntry instead of per whole-entity, this becomes that cluster's actual
    // clusterID with no change to the encoding itself (still the high 32 bits of the combined ID).
    outClusterID = v.meshID;
    // Local TriangleID: this draw is a flat, non-indexed-instance triangle list read via
    // gl_VertexIndex (see the index fetch above), so dividing by 3 recovers the ordinal triangle
    // number within the whole draw's index stream -- "local" to this draw call today, and, once
    // clustered draws land, local to whichever cluster's own [0, kMaxClusterTriangles) index range
    // it will become (see geometry::kMaxClusterTriangles, ClusterFormat.h).
    outTriangleID = uint(gl_VertexIndex) / 3u;

    // 5. Transform position
    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}