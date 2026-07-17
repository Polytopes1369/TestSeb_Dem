#version 460
#extension GL_GOOGLE_include_directive : enable

// Phase 7a (UE5.8 parity roadmap, hero asset tessellation): tessellation control shader --
// screen-space adaptive tessellation, in the same philosophy as ClusterDAGScreenError.comp's own
// pixel-error-targeted Nanite LOD cut (project a world-space measure through the camera, compare
// against a fixed pixel threshold), adapted here to a per-EDGE 2-point screen-space projection
// (not a single error-sphere radius, since a tessellated patch edge has no single proxy sphere the
// way a DAG cluster does). Both endpoints of a shared edge come from the SAME 2 indexed vertices
// (the Fallback Mesh is a proper indexed mesh, not per-triangle-duplicated), so two adjacent
// triangles independently compute the IDENTICAL tess level for their shared edge -- this is what
// keeps the tessellated (and, after .tese's displacement, the DISPLACED) surface crack-free: the
// Vulkan tessellator's own guarantee for matching per-edge outer levels + shared control points.
//
// Tessellation levels are computed from the UNDISPLACED (rest-shape) positions -- a deliberate
// choice, not an oversight: computing them from the displaced surface would require already
// knowing the displacement before deciding how finely to sample it (a circular dependency), and
// this hero asset's displacement amplitude is modest relative to its own silhouette size, so this
// is not a meaningful source of visible under-tessellation.

layout(vertices = 3) out;

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

// Fixed target (this phase's own chosen granularity) -- larger = coarser tessellation for the
// same screen-space edge length. Comfortably conservative for a single hero object.
const float kTargetPixelsPerSegment = 24.0;
// Comfortably under the Vulkan-guaranteed maxTessellationGenerationLevel minimum of 64 -- see
// renderer::HeroTessellationPass's own class comment on why no runtime capability query is needed.
const float kMaxTessLevel = 16.0;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) in vec3 inWorldNormal[];

layout(location = 0) out vec3 outWorldPos[];
layout(location = 1) out vec3 outWorldNormal[];

vec2 ProjectToPixels(vec3 worldPos) {
    vec4 clip = pc.viewProj * vec4(worldPos, 1.0);
    vec2 ndc = clip.xy / max(clip.w, 1.0e-5);
    return (ndc * 0.5 + 0.5) * vec2(pc.viewportWidth, pc.viewportHeight);
}

float EdgeTessLevel(vec3 a, vec3 b) {
    float pixelLength = length(ProjectToPixels(b) - ProjectToPixels(a));
    return clamp(pixelLength / kTargetPixelsPerSegment, 1.0, kMaxTessLevel);
}

void main() {
    outWorldPos[gl_InvocationID] = inWorldPos[gl_InvocationID];
    outWorldNormal[gl_InvocationID] = inWorldNormal[gl_InvocationID];

    if (gl_InvocationID == 0) {
        // gl_TessLevelOuter[i] controls the edge OPPOSITE control point i: Outer[0] = edge(1,2),
        // Outer[1] = edge(2,0), Outer[2] = edge(0,1) -- standard Vulkan/OpenGL triangle-patch
        // convention.
        float outer0 = EdgeTessLevel(inWorldPos[1], inWorldPos[2]);
        float outer1 = EdgeTessLevel(inWorldPos[2], inWorldPos[0]);
        float outer2 = EdgeTessLevel(inWorldPos[0], inWorldPos[1]);
        gl_TessLevelOuter[0] = outer0;
        gl_TessLevelOuter[1] = outer1;
        gl_TessLevelOuter[2] = outer2;
        // Standard average-of-outer inner level -- this is a single isolated hero object, not a
        // shared-edge terrain grid needing a more elaborate inner-level scheme.
        gl_TessLevelInner[0] = (outer0 + outer1 + outer2) / 3.0;
    }
}
