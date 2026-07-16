#pragma once
// Small, constexpr lookup keyed by geometry::ClusterIndexEntry::materialID (itself a verbatim copy
// of core::EntityData::materialID, carried through the whole LOD/cull pipeline -- see
// ClusterCullingPass.h's ClusterCullMetadata::materialID and ClusterLODSelectionPass.h's
// LODNodeMetadata::materialID). This is the RENDERER-side counterpart to
// geometry::EntityMaterialTable.h: that table drives cook-time-only per-cluster fields
// (maxWPOAmplitude/maskTextureIndex, baked into the .cache file); this one drives real PBR shading
// parameters consumed at runtime by ClusterResolve.comp, uploaded once into a GPU SSBO by
// renderer::ClusterResolvePass::Init() (see that class's own comment) -- no on-disk section is
// needed for it, matching this codebase's "bindless == fixed-size compile-time array" convention
// (the exact precedent is renderer::ProceduralMaskGenerator's 64-slot procedural mask array).
//
// Unmapped materialIDs (>= kMaxMaterials) are clamped by ClusterResolve.comp before indexing, so
// they safely fall back to whatever this table's last slot holds.

#include <array>
#include <cstdint>

#include "core/maths/Maths.h"

namespace renderer {

    // GLSL-friendly, std430-compatible mirror of MaterialParams in
    // src/shaders/include/material_params.glsl -- 32 bytes, two {vec3, float} pairs, naturally
    // 16-byte aligned with no explicit padding needed (matches this codebase's own
    // ClusterCullMetadata/LODNodeMetadata field-ordering convention: every vec3 is immediately
    // followed by the scalar that fills its std430 base-alignment gap).
    struct MaterialParameters {
        maths::vec3 baseColor;
        float roughness;
        maths::vec3 emissive;
        float metallic;
    };
    static_assert(sizeof(MaterialParameters) == 32,
        "MaterialParameters must match MaterialParams in material_params.glsl exactly (std430 layout)");

    // Bounds both this CPU-side array and the matching GPU SSBO (ClusterResolvePass allocates
    // exactly kMaxMaterials slots) -- generous headroom over the handful of recipes seeded below,
    // extend this table (not this constant) as new procedural material types are authored.
    inline constexpr uint32_t kMaxMaterials = 32u;

    // How many of kMaxMaterials' slots BuildMaterialParameterTable() actually overrides with a
    // concrete, visually distinct recipe below (indices [0, kAuthoredMaterialRecipeCount)) --
    // exposed so callers assigning materialID to entities (e.g. VulkanContext::BuildEntityData) can
    // cycle through the real recipes instead of a magic number duplicated at the call site.
    inline constexpr uint32_t kAuthoredMaterialRecipeCount = 6u;

    // materialID -> PBR parameters. Every slot starts at a neutral matte-gray default (so any
    // materialID beyond the explicitly authored recipes below still renders as a sane, non-black
    // surface instead of undefined GPU memory), then the first few slots are overridden with
    // concrete, visually distinct recipes covering the demo scene's current primitive types.
    // Index 1 (foliage) intentionally matches geometry::EntityMaterialTable.h's existing case 1u
    // (WPO sway) so the swaying entity also gets a real green material instead of the old
    // clusterID-hashed procedural hue.
    constexpr std::array<MaterialParameters, kMaxMaterials> BuildMaterialParameterTable() {
        std::array<MaterialParameters, kMaxMaterials> table{};
        for (MaterialParameters& params : table) {
            params = MaterialParameters{ maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f };
        }
        // Default / stone.
        table[0] = MaterialParameters{ maths::vec3(0.55f, 0.53f, 0.50f), 0.85f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f };
        // Foliage -- keeps EntityMaterialTable.h's existing WPO sway (materialID == 1u).
        table[1] = MaterialParameters{ maths::vec3(0.20f, 0.45f, 0.12f), 0.75f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f };
        // Metal.
        table[2] = MaterialParameters{ maths::vec3(0.70f, 0.71f, 0.73f), 0.30f, maths::vec3(0.0f, 0.0f, 0.0f), 1.0f };
        // Wood / bark.
        table[3] = MaterialParameters{ maths::vec3(0.35f, 0.22f, 0.12f), 0.70f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f };
        // Emissive / crystal.
        table[4] = MaterialParameters{ maths::vec3(0.10f, 0.05f, 0.05f), 0.40f, maths::vec3(1.2f, 0.30f, 0.05f), 0.0f };
        // Glossy blue (water-like).
        table[5] = MaterialParameters{ maths::vec3(0.05f, 0.20f, 0.35f), 0.10f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f };
        return table;
    }

    inline constexpr std::array<MaterialParameters, kMaxMaterials> kMaterialParameterTable = BuildMaterialParameterTable();

}
