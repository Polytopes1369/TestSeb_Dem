#ifndef MEGALIGHTS_BVH_GLSL
#define MEGALIGHTS_BVH_GLSL

// Feature 1 of Phase 4 (MegaLights advanced roadmap: light BVH for RIS spatial bias, temporal
// ReSTIR with revalidated visibility -- see the approved plan). GPU-side traversal of
// geometry::LightBVH (see that header's own "why a separate, duplicated implementation" comment).
//
// NOT reducing SelectLightRIS's candidate draw to something cheaper than O(1)/pixel -- there is no
// O(N) cost to accelerate here: MegaLightsShade.comp already draws a fixed
// kMegaLightsCandidateCount population regardless of total light count (see megalights_ris.glsl's
// own header comment). This instead biases WHICH candidates that fixed-size draw pulls from,
// toward lights actually near the current shading point -- the same role real UE5.8 MegaLights'
// own spatial structure plays.
//
// The caller must #include "math_utils.glsl" (for AABBOverlapsAABB) and "megalights_types.glsl"
// (for kMegaLightsSpatialPoolCapacity) before this include, AND define MEGALIGHTS_BVH_SET/
// MEGALIGHTS_BVH_NODES_BINDING/MEGALIGHTS_BVH_INDICES_BINDING -- same caller-defined-binding-macro
// convention as megalights_ris.glsl's own MEGALIGHTS_LIGHTS_SET/MEGALIGHTS_LIGHTS_BINDING.
//
// --- Stack-based traversal ---
// Mirrors SDFRayMarch.comp's own TraceBVH exactly in shape: a fixed-size stack, the same
// left-child-always-(nodeIndex+1)/leftFirst-holds-right-child-index-or-leaf-offset convention (see
// geometry::LightBVHNode's own layout comment) -- but prunes by a plain AABB-overlap test against
// a static `worldPos +/- biasRadius` query box instead of a ray-AABB slab test, since there is no
// ray to walk here: every light node whose own (radius-expanded) AABB intersects that box is a
// candidate, full stop.

#ifndef MEGALIGHTS_BVH_SET
#error "Define MEGALIGHTS_BVH_SET/MEGALIGHTS_BVH_NODES_BINDING/MEGALIGHTS_BVH_INDICES_BINDING before including megalights_bvh.glsl"
#endif

// Byte-for-byte mirror of geometry::LightBVHNode (LightBVH.h) -- float[3] arrays (NOT vec3) for
// boundsMin/boundsMax, exactly matching SDFRayMarch.comp's own BVHNode declaration for the
// identical reason: a GLSL vec3 member forces 16-byte base alignment inside a std430 struct, which
// would silently desynchronize this declaration from the CPU struct's real 32-byte packed layout.
struct LightBVHNode {
    float boundsMin[3];
    int leftFirst; // Interior (count == 0): left child is ALWAYS nodeIndex + 1; this field holds the RIGHT child's node index instead. Leaf (count > 0): first index into leafLightIndices.
    float boundsMax[3];
    int count;     // 0 == interior node; > 0 == leaf, holding this many lights starting at leftFirst.
};
layout(std430, set = MEGALIGHTS_BVH_SET, binding = MEGALIGHTS_BVH_NODES_BINDING) readonly buffer LightBVHNodesBuffer {
    LightBVHNode bvhNodes[];
};
layout(std430, set = MEGALIGHTS_BVH_SET, binding = MEGALIGHTS_BVH_INDICES_BINDING) readonly buffer LightBVHIndicesBuffer {
    uint leafLightIndices[];
};

// Walks geometry::LightBVH from its root (node 0 -- BuildLightBVH's own build order guarantees
// node 0 is always the root when the tree is non-empty, see LightBVH.h's own comment), pruning by
// AABB-overlap against a `worldPos +/- biasRadius` query box, greedily filling
// `outPool[0..outPoolCount)` with up to kMegaLightsSpatialPoolCapacity light indices. RIS already
// treats its own candidate population as approximate (see megalights_ris.glsl's own SelectLightRIS
// comment), so this need not be an exhaustive intersection list -- "close enough, cheaply found" is
// the whole point. `outPoolCount` is left at 0 (an empty pool) when the BVH itself is empty
// (bvhNodes.length() == 0u, i.e. zero lights at renderer::MegaLightsPass::Init time) or the query
// box overlaps no leaf at all -- SelectLightRIS's own caller falls back to its full-population draw
// in that case, so an empty pool here is always a safe, non-fatal result.
void GatherSpatialLightCandidates(vec3 worldPos, float biasRadius, out uint outPool[kMegaLightsSpatialPoolCapacity], out uint outPoolCount) {
    outPoolCount = 0u;

    if (bvhNodes.length() == 0u) {
        return;
    }

    vec3 queryMin = worldPos - vec3(biasRadius);
    vec3 queryMax = worldPos + vec3(biasRadius);

    // Fixed-size stack: geometry::LightBVH's own median-split build keeps the tree shallow
    // (depth ~= log2(lightCount / kLightBVHMaxLightsPerLeaf)), so 32 entries is generous headroom
    // even for renderer::kMaxMegaLights' worth of lights -- same sizing rationale as
    // SDFRayMarch.comp's own TraceBVH stack.
    int stack[32];
    int sp = 0;
    stack[sp++] = 0; // Root is always node 0 (see this function's own comment).

    while (sp > 0 && outPoolCount < kMegaLightsSpatialPoolCapacity) {
        int nodeIdx = stack[--sp];
        LightBVHNode node = bvhNodes[nodeIdx];
        vec3 bmin = vec3(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
        vec3 bmax = vec3(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);

        if (!AABBOverlapsAABB(queryMin, queryMax, bmin, bmax)) {
            continue;
        }

        if (node.count > 0) {
            for (int k = 0; k < node.count && outPoolCount < kMegaLightsSpatialPoolCapacity; ++k) {
                outPool[outPoolCount] = leafLightIndices[node.leftFirst + k];
                outPoolCount++;
            }
        } else if (sp < 30) { // Headroom guard -- see stack[] declaration comment.
            stack[sp++] = nodeIdx + 1;    // Left child (see LightBVHNode's own layout comment).
            stack[sp++] = node.leftFirst; // Right child.
        }
    }
}

#endif
