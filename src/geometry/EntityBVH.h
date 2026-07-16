#pragma once
// CPU-built spatial acceleration structure over entity world-space AABBs (see geometry::
// FallbackMeshIndexEntry -- entities are static in this engine, so local space already IS world
// space, see renderer::GlobalSDFPass's own "Scope" note), consumed by renderer::SDFRayMarchPass's
// compute shader to narrow a coarse Global SDF hit down to the handful of per-entity Mesh SDFs
// actually worth sampling precisely near that point, instead of testing every entity in the scene.
//
// --- Why a CPU-built BVH, not a GPU acceleration structure ---
// This is a flattened, GPU-readable binary BVH built once on the CPU at load time and uploaded as
// a plain read-only storage buffer -- NOT a VK_KHR_acceleration_structure (this codebase's Vulkan
// context does not currently enable that extension; see VulkanContext.cpp's own deviceExtensions
// list). A compute shader traverses it with a manual stack-based walk (see SDFRayMarch.comp) --
// exactly the "table de structures d'accélération CPU" a Lumen-style ray march needs to identify
// candidate Mesh SDFs near the current marching position, without requiring hardware ray tracing
// acceleration structures at all.
//
// --- Layout: flattened binary tree, depth-first pre-order ---
// Node 0 is the root. For an INTERIOR node (count == 0), the left child is ALWAYS at (this node's
// own index + 1) -- guaranteed by construction (BuildEntityBVH appends the entire left subtree
// immediately after its parent, before the right subtree is appended) -- and leftFirst instead
// holds the RIGHT child's node index. For a LEAF node (count > 0), leftFirst is the starting
// offset into EntityBVH::entityIndices, which holds `count` consecutive original-list indices.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"

namespace geometry {

    // How many entities a leaf holds before splitting further -- small enough that a compute
    // shader's leaf-entity loop (see SDFRayMarch.comp) stays cheap, large enough that a scene with
    // only a few entities does not build a needlessly deep tree of near-empty nodes.
    constexpr uint32_t kBVHMaxEntitiesPerLeaf = 4u;

    // Flattened, GPU-uploadable BVH node -- exactly 32 bytes, matching this codebase's existing
    // flat-scalar-field convention for CPU/GPU struct pairs (see renderer::GlobalSDFCompositePC's
    // own comment) so std430 layout is unambiguous once uploaded as a storage buffer.
    struct BVHNode {
        float boundsMin[3];
        int32_t leftFirst; // Interior: right child's node index (left is always index+1). Leaf: entityIndices start offset.
        float boundsMax[3];
        int32_t count;      // 0 = interior node; > 0 = leaf, holding this many entities starting at leftFirst.
    };
    static_assert(sizeof(BVHNode) == 32, "BVHNode must stay a flat 32 bytes for direct GPU upload");

    // A CPU-built BVH over a fixed set of entity AABBs, ready for direct GPU upload.
    struct EntityBVH {
        std::vector<BVHNode> nodes;          // nodes[0] is the root; empty iff the input entity list was empty.
        std::vector<uint32_t> entityIndices; // Flattened leaf contents -- indices into the ORIGINAL
                                              // input vector passed to BuildEntityBVH (i.e.
                                              // entityIndices[k] is an index into that vector, NOT
                                              // an entityID itself; the caller derives the entityID
                                              // via originalEntities[entityIndices[k]].entityID).
    };

    // Builds a median-split (top-down, largest-centroid-extent-axis) binary BVH over `entities`'
    // world-space AABBs (geometry::FallbackMeshIndexEntry::boundsMin/boundsMax -- already world
    // space, see class comment). Returns an EntityBVH with empty nodes/entityIndices for an empty
    // `entities` input (a valid, no-op BVH: SDFRayMarch.comp's traversal simply finds no
    // candidates).
    EntityBVH BuildEntityBVH(const std::vector<FallbackMeshIndexEntry>& entities);

}
