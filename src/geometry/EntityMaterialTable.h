#pragma once
// Small, constexpr lookup keyed by core::EntityData::materialID (see core/EntityData.h -- the
// field already existed, unused by any renderer code, until this table). Drives two per-entity
// values stamped uniformly into every DAG level of that entity's clusters (see
// VirtualGeometryCacheTest.cpp's per-entity loop and geometry::ClusterIndexEntry's matching
// fields): the maximum World Position Offset sway amplitude the vertex shaders may apply, and
// which slot of the bindless procedural cutout mask array (mask_sampling.glsl) to sample.
//
// Unmapped materialIDs fall back to {0.0f, kInvalidMaskTextureIndex} -- no sway, no cutout -- so
// any entity not explicitly listed here renders exactly as it did before this table existed.

#include <cstdint>
#include "geometry/ClusterFormat.h"
// Reserved tree bark/foliage materialID constants only (kTreeBarkMaterialID/kTreeLeafMaterialID) --
// a lightweight, dependency-free constants header, same "geometry may reference a plain renderer
// data/constant header" precedent already established by ClusterDAG.h's own
// #include "renderer/RenderTypes.h".
#include "renderer/MaterialParameterTable.h"

namespace geometry {

    struct EntityMaterialProperties {
        float maxWPOAmplitude;
        uint32_t maskTextureIndex;
    };

    // materialID -> {maxWPOAmplitude, maskTextureIndex}. Extend this switch as new procedurally-
    // generated material types (foliage, tree canopies, etc.) are added to the scene.
    //
    // renderer::ProceduralTreePass (real tree geometry, not just a foliage-look material) wires its
    // wind sway through this SAME existing mechanism -- maxWPOAmplitude feeds
    // src/shaders/include/wpo_deformation.glsl's ApplyWPODeformation, a per-cluster, height-scaled,
    // phase-offset sine sway (see that file's own header comment). No real Atmos wind-VECTOR is
    // plumbed into WPO anywhere in this codebase today (confirmed by grepping every
    // SampleWindVelocity/wind-direction call site: all of them feed AtmosClouds/AtmosVolumetricFog/
    // ParticleSimulation, never ClusterRaster.vert or this file) -- so this IS this project's
    // documented time-based-sine placeholder (per the tree-generation task's own explicit fallback
    // allowance), not a simplification introduced here. A future pass wiring a real wind vector into
    // WPO would extend ApplyWPODeformation's signature, not this table.
    constexpr EntityMaterialProperties GetEntityMaterialProperties(uint32_t materialID) {
        switch (materialID) {
            case 1u: return EntityMaterialProperties{ 0.15f, 0u }; // Example foliage/tree material: sways, cutout mask slot 0.
            // Tree bark/branches: solid cylinder geometry (no cutout mask), a small sway so the
            // whole tree isn't perfectly rigid -- much less than the foliage's own sway below, since
            // a real trunk/branch is far stiffer than a leaf cluster.
            case renderer::kTreeBarkMaterialID: return EntityMaterialProperties{ 0.06f, kInvalidMaskTextureIndex };
            // Tree foliage: the leaf cross-quad cards -- full opacity-cutout (mask slot 0, the same
            // procedurally-generated blotchy leaf-cluster silhouette every other foliage-style
            // material already reuses, see ProceduralMaskGenerate.comp's own header comment) plus
            // the strongest sway in this table (a leaf cluster is the most wind-responsive part of a
            // tree).
            case renderer::kTreeLeafMaterialID: return EntityMaterialProperties{ 0.35f, 0u };
            // 10-tree-species scene: per-species foliage variants -- same cutout mask slot 0 +
            // wind-sway mechanism as kTreeLeafMaterialID above, with per-species sway amplitude
            // (a willow's trailing fronds move the most, a dead tree's dry remnants barely at all).
            case renderer::kTreeLeafPineMaterialID: return EntityMaterialProperties{ 0.25f, 0u };
            case renderer::kTreeLeafBirchMaterialID: return EntityMaterialProperties{ 0.40f, 0u };
            case renderer::kTreeLeafAutumnMaterialID: return EntityMaterialProperties{ 0.40f, 0u };
            case renderer::kTreeLeafWillowMaterialID: return EntityMaterialProperties{ 0.50f, 0u };
            case renderer::kTreeLeafDeadMaterialID: return EntityMaterialProperties{ 0.15f, 0u };
            // Per-species bark variants -- solid cylinder geometry like kTreeBarkMaterialID above
            // (no cutout mask); dead wood is stiffer than living bark, so it sways even less.
            case renderer::kTreeBarkBirchMaterialID: return EntityMaterialProperties{ 0.06f, kInvalidMaskTextureIndex };
            case renderer::kTreeBarkDeadMaterialID: return EntityMaterialProperties{ 0.03f, kInvalidMaskTextureIndex };
            default: return EntityMaterialProperties{ 0.0f, kInvalidMaskTextureIndex };
        }
    }

}
