#pragma once
// UE5.8-parity gap G1 (Decal system): a CPU-built spatial acceleration structure over the
// world-space AABBs of every projected decal's oriented box volume, consumed by
// renderer::DecalProjectionPass's compute shader (DecalProject.comp) to narrow the per-pixel
// "which decals cover this reconstructed world position" test down to the handful of decal boxes
// actually near that point, instead of testing every decal in the scene per pixel.
//
// --- Why a SEPARATE, duplicated implementation from geometry::EntityBVH / geometry::LightBVH ---
// Byte-for-byte identical flattened 32-byte node layout and the same top-down median-split/flatten
// build algorithm as geometry::EntityBVH (SDF ray march) and geometry::LightBVH (MegaLights RIS) --
// see EntityBVH.h's own header comment for the full layout derivation (node 0 is always the root, an
// interior node's left child is always index+1 with leftFirst holding the right child's index
// instead, a leaf's leftFirst is a start offset into a flattened index array). Deliberately NOT
// generalized into a shared base/template, for the exact same reason LightBVH.h documents for itself:
// each of the three BVHs is hard-bound to a differently-shaped AABB source (a precomputed min/max
// pair for entities, a radius-expanded point for lights, an oriented-box's 8-corner span here), and a
// generic templated interface none of the three callers otherwise needs would leak each feature's
// specifics into a shared abstraction. The CPU build cost here is trivial (a handful of decals, once,
// at renderer::DecalProjectionPass::Init time), so duplication costs nothing at runtime and keeps
// this entirely isolated from the already-shipped SDF-ray-march and MegaLights BVH paths.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"

namespace geometry {

    // Mirrors geometry::kBVHMaxEntitiesPerLeaf (EntityBVH.h) / geometry::kLightBVHMaxLightsPerLeaf
    // (LightBVH.h) -- see those constants' own comments for the leaf-size trade-off rationale,
    // identical here (small enough that a compute shader's leaf loop stays cheap, large enough that a
    // sparse decal population does not build a needlessly deep tree of near-empty nodes).
    constexpr uint32_t kDecalBVHMaxDecalsPerLeaf = 4u;

    // Flattened, GPU-uploadable BVH node -- byte-for-byte identical FIELD LAYOUT to geometry::BVHNode
    // (EntityBVH.h) / geometry::LightBVHNode (LightBVH.h), but declared as its own distinct type per
    // this file's own header comment on why the three BVHs are deliberately NOT unified.
    struct DecalBVHNode {
        float boundsMin[3];
        int32_t leftFirst; // Interior: right child's node index (left is always index+1). Leaf: decalIndices start offset.
        float boundsMax[3];
        int32_t count;      // 0 = interior node; > 0 = leaf, holding this many decals starting at leftFirst.
    };
    static_assert(sizeof(DecalBVHNode) == 32, "DecalBVHNode must stay a flat 32 bytes for direct GPU upload");

    // A single decal's precomputed world-space AABB -- the build input, mirroring how LightBVH derives
    // an AABB from position +/- radius (there computed inside the builder; here supplied precomputed by
    // the caller, since an oriented box's world AABB is derived from its transform, not from any single
    // field of the GPU decal struct).
    struct DecalBoxBounds {
        maths::vec3 boundsMin{};
        maths::vec3 boundsMax{};
    };

    // A CPU-built BVH over a fixed set of decal box AABBs, ready for direct GPU upload.
    struct DecalBVH {
        std::vector<DecalBVHNode> nodes;    // nodes[0] is the root; empty iff the input decal count was zero.
        std::vector<uint32_t> decalIndices; // Flattened leaf contents -- indices DIRECTLY into the
                                            // `bounds` array passed to BuildDecalBVH, matching the GPU
                                            // decal SSBO's own g_Decals.decals[] indexing exactly (no
                                            // extra ID indirection layer, same convention as
                                            // LightBVH::lightIndices).
    };

    // Builds a median-split (top-down, largest-centroid-extent-axis) binary BVH over `bounds[0,
    // count)`. Returns a DecalBVH with empty nodes/decalIndices when count == 0 -- a valid, no-op
    // BVH: the shader's traversal simply finds no candidates and the decal pass leaves every pixel
    // untouched.
    DecalBVH BuildDecalBVH(const DecalBoxBounds* bounds, uint32_t count);

}
