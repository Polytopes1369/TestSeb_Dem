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

namespace geometry {

    struct EntityMaterialProperties {
        float maxWPOAmplitude;
        uint32_t maskTextureIndex;
    };

    // materialID -> {maxWPOAmplitude, maskTextureIndex}. Extend this switch as new procedurally-
    // generated material types (foliage, tree canopies, etc.) are added to the scene.
    constexpr EntityMaterialProperties GetEntityMaterialProperties(uint32_t materialID) {
        switch (materialID) {
            case 1u: return EntityMaterialProperties{ 0.15f, 0u }; // Example foliage/tree material: sways, cutout mask slot 0.
            default: return EntityMaterialProperties{ 0.0f, kInvalidMaskTextureIndex };
        }
    }

}
