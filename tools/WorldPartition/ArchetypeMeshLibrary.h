#pragma once
// Phase 5 (Streaming & Monde roadmap, Part 2, Gap 1): offline-only, zero-Vulkan-dependency CPU mesh
// constructors for this demo's 4 fixed streaming archetypes (Rock/Bush/Tree/Debris -- see
// world::kArchetypeShapeCount and renderer::VulkanContext::GenerateGeometry()'s own streaming-pool
// switch statement, both of which this file's shape indices are ordered to match EXACTLY: 0 = Rock
// (icosphere), 1 = Bush (UV sphere), 2 = Tree (capsule), 3 = Debris (torus)).
//
// Why this exists: HlodPipeline::GatherCellMeshes needs a real geometry::SimplifiableMesh per
// authored actor (ActorMeshFetchFn) to build a genuine, simplifiable HLOD proxy -- BakeDemoWorld.cpp
// previously had no mesh source at all (an offline tool has zero access to the runtime's GPU compute
// shaders, e.g. geom_icosphere.comp, that bake these same shapes for the shipping executable). This
// is a from-scratch, native C++23 CPU re-implementation of the same 4 silhouettes at a
// simplification-appropriate triangle count (dense enough that geometry::SimplifyMeshQEM has real
// reduction work to do collapsing down to a small per-cell HLOD triangle budget, e.g. 64
// triangles) -- it does not need to be byte-identical to the GPU generator's own vertex ordering,
// only geometrically equivalent (same shape, same approximate radius), since the two are consumed by
// completely different pipelines (this one feeds an offline simplifier; the GPU one feeds the live
// Nanite cluster DAG for the "fine" variant).
//
// Every mesh is generated in LOCAL space, centered at its own local origin (0,0,0) -- exactly the
// same "baked at a parking position, repositioned via a runtime translation" convention
// GenerateGeometry()'s existing streaming-pool block already uses for the fine-variant archetypes
// (see that block's own header comment), so the resulting HLOD proxy can be positioned at an
// arbitrary cell's world position the same way.

#include "geometry/MeshSimplifier.h"

namespace worldpartition {

    // Number of distinct archetype shapes this library provides -- mirrors world::kArchetypeShapeCount
    // (src/world/WorldCellStreamingLoader.h) exactly; kept as its own constant here (not shared,
    // matching this codebase's established src//tools/ type-duplication convention, see
    // StreamingTypes.h's own header comment) since tools/ has zero compile-time dependency on src/world/.
    inline constexpr uint32_t kArchetypeShapeCount = 4;

    // Builds one archetype's CPU mesh, indexed exactly like renderer::VulkanContext::GenerateGeometry()'s
    // streaming-pool switch (shape % kArchetypeShapeCount): 0 = Rock, 1 = Bush, 2 = Tree, 3 = Debris.
    // Every returned mesh has `locked` sized to match `positions` and entirely false (a freshly built
    // archetype mesh has no pre-existing neighbor to stay crack-free against -- unlike
    // ClusterGrouping's own boundary-locked groups) and `uvs` populated (simple planar/spherical
    // projection per shape, carried through QEM simplification via SimplifiableMesh's own documented
    // midpoint-averaging convention).
    geometry::SimplifiableMesh BuildArchetypeMesh(uint32_t archetypeShape);

}
