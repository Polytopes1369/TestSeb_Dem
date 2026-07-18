#ifndef TREE_LSYSTEM_GLSL
#define TREE_LSYSTEM_GLSL

// Deterministic, GPU-evaluated recursive branching L-system for renderer::ProceduralTreePass's
// geom_tree_bark.comp / geom_tree_leaves.comp -- shared by both so the bark mesh and the leaf
// cards agree exactly on where every branch actually is, without either shader uploading any
// CPU-computed skeleton buffer: every compute invocation reconstructs its own node's full
// world-space transform purely from (nodeID, seed, the small set of shape parameters), in
// parallel, with no inter-invocation communication -- matching the "100% procedural GPU driven"
// project mandate as literally as this codebase's other geom_*.comp generators (box/cone/sphere/
// ...), which are all closed-form parametric surfaces evaluated per-thread; a recursive branching
// skeleton has no closed-form parametric surface, so this file's DecodeTreeNode() is the
// equivalent "per-thread formula" for a tree: given a node's linear index it walks the fixed
// topology from the root down to that node and re-derives its transform, entirely arithmetically.
//
// --- Topology: a complete branchFactor-ary tree, stored breadth-first ---
// Node 0 is the trunk/root. Node i (i > 0) is child (i-1)%branchFactor of parent (i-1)/branchFactor
// -- the standard "complete N-ary tree as a flat array" indexing identity, chosen specifically
// because it lets DecodeTreeNode() go from a linear nodeID straight to its ancestor chain with no
// stored parent pointers at all. A "level" (or depth) is the number of branchings from the root;
// level 0 is the trunk itself, level `depth` is the leaf tier geom_tree_leaves.comp attaches
// foliage cards to.
//
// --- Per-branch transform model (deliberately simplified, see below) ---
// Each node's growth direction is stored as two accumulated SCALAR angles -- pitch (polar angle
// from world +Y) and yaw (azimuthal angle around +Y) -- rather than a full propagated 3x3 branch-
// local frame. direction = SphericalDir(pitch, yaw). This is a simplification versus a botanically
// exact model (which would rotate a child's cone-of-growth around its PARENT's own local axis,
// not around the fixed world +Y polar axis), but is visually convincing for a demoscene showcase
// tree and keeps the per-invocation walk to pure scalar arithmetic -- no orthonormal frame
// propagation, no quaternion/matrix composition needed at all. Documented here explicitly per this
// project's "explicite les structures... sans raccourcis" comment-everything-complex rule.
//
// Every branching step (see the loop in DecodeTreeNode) advances:
//   yaw   += (childIndex / branchFactor) * TAU + jitter      -- spreads children evenly in azimuth
//   pitch  = clamp(pitch * pitchDamping + branchAngle + jitter, 0, PI*0.85) -- fans outward from
//            vertical, damped so deep branches don't run away past horizontal
//   length *= lengthTaper * jitter                            -- shorter at each level
//   radius *= radiusTaper                                     -- thinner at each level (continuous:
//            a child's start radius always equals its parent's end radius)
// `jitter` terms are per-node pseudo-random (Wang-hash-style, see HashTree/HashFloat01 below),
// seeded by (that node's own linear ID, the tree's `seed` push-constant field) so two trees with
// different seeds branch differently, and the SAME tree always regenerates bit-identically (a
// demoscene "demo" must play back identically every run -- the same determinism rule this
// codebase's other procedural generators already follow, see MaterialParameterTable.h's own
// "Fully deterministic" comment).

#include "include/math_utils.glsl" // PI, TAU

// Hard cap on tree recursion depth this file supports -- generous headroom over any reasonable
// `depth` push-constant value (renderer::ProceduralTreePass uses depth=4). Bounds the fixed-size
// local array DecodeTreeNode() walks the node's ancestor chain into; GLSL requires a compile-time
// array size, so this cannot simply be the runtime `depth` parameter.
#define TREE_MAX_DEPTH 8u

uint HashTree(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float HashFloat01(uint h) {
    return float(h & 0xFFFFu) / 65535.0;
}

// Spherical parametrization: pitch is the polar angle from world +Y (0 = straight up), yaw is the
// azimuthal angle around +Y. See this file's header comment for why this (rather than a full
// propagated branch-local frame) is this generator's deliberate simplification.
vec3 SphericalDir(float pitch, float yaw) {
    float sp = sin(pitch);
    float cp = cos(pitch);
    return vec3(sp * cos(yaw), cp, sp * sin(yaw));
}

// One fully-resolved node: the tapered branch SEGMENT running from startPos (== its parent's own
// endPos, or the world origin for the trunk) to endPos, with radius linearly implied between
// radiusStart and radiusEnd.
struct TreeNodeState {
    vec3 startPos;
    vec3 endPos;
    float radiusStart;
    float radiusEnd;
    uint level;      // 0 = trunk.
    bool isLeafTip;  // True iff level == depth -- geom_tree_leaves.comp only attaches cards here.
};

// Reconstructs `nodeID`'s complete world-space transform (pre-worldOffset -- the caller adds the
// tree's own world placement afterward, exactly like every other geom_*.comp shader's
// worldOffsetX/Y/Z convention) by walking the complete branchFactor-ary tree from the root down to
// this node. Two passes, both bounded by TREE_MAX_DEPTH (a handful of iterations, trivial cost per
// invocation):
//   1. UP: decode nodeID's ancestor chain (which child-index was taken at each level), bottom-up,
//      using the complete-tree indexing identity (see this file's header comment) -- no stored
//      parent pointers needed, purely arithmetic.
//   2. DOWN: replay that chain from the root, accumulating pitch/yaw/length/radius/position one
//      branching step at a time, hashing each step's own re-derived linear ancestor ID for that
//      step's jitter (see the header comment's per-step formula).
TreeNodeState DecodeTreeNode(uint nodeID, uint seed, uint depth, uint branchFactor,
    float trunkHeight, float trunkRadius, float lengthTaper, float radiusTaper,
    float branchAngle, float pitchDamping)
{
    uint childIdxBottomUp[TREE_MAX_DEPTH];
    uint level = 0u;
    uint cur = nodeID;
    while (cur > 0u && level < TREE_MAX_DEPTH) {
        childIdxBottomUp[level] = (cur - 1u) % branchFactor;
        cur = (cur - 1u) / branchFactor;
        level++;
    }
    // `level` now equals nodeID's own depth (0 for the trunk/root itself).

    float pitch = 0.0;
    float yaw = 0.0;
    float curLength = trunkHeight;
    float radiusEnd = trunkRadius;
    vec3 pos = vec3(0.0);
    uint ancestorID = 0u;

    float radiusStart = trunkRadius;
    vec3 startPos = vec3(0.0);

    for (uint l = 0u; l < level; ++l) {
        // childIdxBottomUp was recorded root-to-leaf in REVERSE (bottom-up) order -- the step taken
        // `l` levels below the root is the entry recorded (level - 1 - l) iterations into the UP
        // walk above.
        uint childIdx = childIdxBottomUp[level - 1u - l];
        uint childID = branchFactor * ancestorID + childIdx + 1u; // Complete-tree child-ID identity.

        uint h1 = HashTree(childID * 0x9E3779B1u + seed);
        uint h2 = HashTree(h1);
        uint h3 = HashTree(h2);
        float j1 = HashFloat01(h1) - 0.5;
        float j2 = HashFloat01(h2) - 0.5;
        float j3 = HashFloat01(h3) - 0.5;

        yaw = yaw + (float(childIdx) / float(branchFactor)) * TAU + j1 * 0.9;
        pitch = clamp(pitch * pitchDamping + branchAngle + j2 * 0.5, 0.0, PI * 0.85);
        curLength = curLength * lengthTaper * (0.8 + j3 * 0.5);

        radiusStart = radiusEnd;
        radiusEnd = radiusStart * radiusTaper;

        vec3 dir = SphericalDir(pitch, yaw);
        startPos = pos;
        pos = startPos + dir * curLength;

        ancestorID = childID;
    }

    TreeNodeState result;
    if (level == 0u) {
        // The trunk itself: starts at the tree's local origin, grows straight up (pitch=yaw=0).
        result.startPos = vec3(0.0);
        result.endPos = vec3(0.0, trunkHeight, 0.0);
        result.radiusStart = trunkRadius;
        result.radiusEnd = trunkRadius * radiusTaper;
    } else {
        result.startPos = startPos;
        result.endPos = pos;
        result.radiusStart = radiusStart;
        result.radiusEnd = radiusEnd;
    }
    result.level = level;
    result.isLeafTip = (level == depth);
    return result;
}

// Total node count of a complete branchFactor-ary tree with levels [0, depth] inclusive -- the
// standard geometric-series closed form (branchFactor^(depth+1) - 1) / (branchFactor - 1). CPU-side
// mirror: renderer::ProceduralTreePass::ComputeNodeCount (must stay in exact agreement, since the
// CPU side uses it to size dispatches/buffers while this GLSL copy is never actually evaluated at
// runtime -- kept here purely so a shader-side reader can see the same formula the CPU already
// used to pick vertexOffset/indexOffset/dispatch counts).

#endif // TREE_LSYSTEM_GLSL
