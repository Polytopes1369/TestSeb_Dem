#pragma once
// Feature 1 of Phase 4 (MegaLights advanced roadmap: light BVH for RIS spatial bias, temporal
// ReSTIR with revalidated visibility -- see the approved plan). A CPU-built spatial acceleration
// structure over renderer::MegaLight AABBs (shape-aware as of UE5.8-parity gap G3: an isotropic
// point/photometric light still bounds as position +/- radius, but a spot light bounds its actual
// cone and a rect light its actual illuminated slab -- see BuildLightBVH's own ComputeLightAABB for
// the exact per-type derivation), consumed by
// megalights_bvh.glsl's GPU-side GatherSpatialLightCandidates to bias MegaLightsShade.comp's
// SelectLightRIS candidate draw toward lights actually near the current shading point. This is
// NOT an O(N)-iteration-avoidance structure -- SelectLightRIS already draws a fixed O(1) budget of
// candidates per pixel regardless of light count (see megalights_ris.glsl's own header comment) --
// it only changes WHICH lights that fixed-size draw is likely to land on, the same role real
// UE5.8 MegaLights' own spatial structure plays.
//
// --- Why a SEPARATE, duplicated implementation from geometry::EntityBVH ---
// Same flattened 32-byte node layout and top-down median-split/flatten build algorithm as
// geometry::EntityBVH (see EntityBVH.h's own header comment for the full layout derivation:
// node 0 is always the root, an interior node's left child is always index+1 with leftFirst
// holding the right child's index instead, a leaf's leftFirst is a start offset into a flattened
// index array) -- deliberately NOT generalized into a shared base/template with EntityBVH.
// EntityBVH is hard-bound to geometry::FallbackMeshIndexEntry (a precomputed min/max AABB pair)
// with exactly one existing caller (renderer::SDFRayMarchPass); forcing it to also serve
// renderer::MegaLight's differently-shaped AABB source (a radius-expanded point, computed here,
// not read off the input type) would either leak MegaLights-specific logic into a shared "SDF ray
// march scene structure" abstraction that has nothing to do with light sampling, or require a
// generic templated interface neither caller actually needs otherwise. The CPU build cost here is
// trivial (~256 lights, once, at renderer::MegaLightsPass::Init time) -- duplication costs nothing
// at runtime and keeps this entirely isolated from Phases 1-3's already-shipped SDF ray march path.
// Matches this codebase's own established precedent for this exact trade-off: see
// megalights_ris.glsl's own header comment on why its trace functions are duplicated per-shader
// rather than shared, for the identical rationale applied GPU-side.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "renderer/MegaLightsTypes.h"

namespace geometry {

    // Mirrors geometry::kBVHMaxEntitiesPerLeaf (EntityBVH.h) -- see that constant's own comment for
    // the leaf-size trade-off rationale, identical here (small enough that a compute shader's leaf
    // loop stays cheap, large enough that a sparse light population does not build a needlessly
    // deep tree of near-empty nodes).
    constexpr uint32_t kLightBVHMaxLightsPerLeaf = 4u;

    // Flattened, GPU-uploadable BVH node -- byte-for-byte identical FIELD LAYOUT to
    // geometry::BVHNode (EntityBVH.h), but declared as its own distinct type (not reused from
    // EntityBVH.h) per this file's own header comment on why the two BVHs are deliberately NOT
    // unified into one shared type.
    struct LightBVHNode {
        float boundsMin[3];
        int32_t leftFirst; // Interior: right child's node index (left is always index+1). Leaf: lightIndices start offset.
        float boundsMax[3];
        int32_t count;      // 0 = interior node; > 0 = leaf, holding this many lights starting at leftFirst.
    };
    static_assert(sizeof(LightBVHNode) == 32, "LightBVHNode must stay a flat 32 bytes for direct GPU upload");

    // A CPU-built BVH over a fixed set of renderer::MegaLight AABBs, ready for direct GPU upload.
    struct LightBVH {
        std::vector<LightBVHNode> nodes;    // nodes[0] is the root; empty iff the input light count was zero.
        std::vector<uint32_t> lightIndices; // Flattened leaf contents -- indices DIRECTLY into the
                                             // `lights` array passed to BuildLightBVH, matching
                                             // MegaLightsSSBO's own g_Lights.lights[] indexing in
                                             // megalights_ris.glsl exactly -- no extra ID
                                             // indirection layer, unlike EntityBVH::entityIndices
                                             // (which maps back through FallbackMeshIndexEntry's own
                                             // entityID field).
    };

    // Builds a median-split (top-down, largest-centroid-extent-axis) binary BVH over
    // `lights[0, lightCount)`'s world-space AABBs, each derived shape-awarely from the light's own
    // type/position/radius/direction/cone/rect fields (G3 -- see the .cpp's own ComputeLightAABB;
    // confirmed fields on renderer::MegaLight, see MegaLightsTypes.h). Returns a
    // LightBVH with empty nodes/lightIndices when lightCount == 0 -- a valid, no-op BVH:
    // GatherSpatialLightCandidates (megalights_bvh.glsl) simply finds no candidates, and
    // SelectLightRIS falls back to its own full-population draw (see megalights_ris.glsl's own
    // comment).
    LightBVH BuildLightBVH(const renderer::MegaLight* lights, uint32_t lightCount);

}
