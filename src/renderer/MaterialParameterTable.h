#pragma once
// Small, constexpr-sized lookup keyed by geometry::ClusterIndexEntry::materialID (itself a verbatim
// copy of core::EntityData::materialID, carried through the whole LOD/cull pipeline -- see
// ClusterCullingPass.h's ClusterCullMetadata::materialID and ClusterLODSelectionPass.h's
// LODNodeMetadata::materialID). This is the RENDERER-side counterpart to
// geometry::EntityMaterialTable.h: that table drives cook-time-only per-cluster fields
// (maxWPOAmplitude/maskTextureIndex, baked into the .cache file); this one drives real PBR shading
// parameters consumed at runtime by ClusterResolve.comp/ClusterResolveBinned.comp/
// TransparentForward.frag, uploaded once into a GPU SSBO by renderer::ClusterResolvePass::Init().
//
// Unmapped materialIDs (>= kMaxMaterials) are clamped by the consuming shaders before indexing, so
// they safely fall back to whatever this table's last slot holds.

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#include "core/maths/Maths.h"

namespace renderer {

    // GLSL-friendly, std430-compatible mirror of MaterialParams in
    // src/shaders/include/material_params.glsl -- 48 bytes, three {vec3, float}-shaped 16-byte
    // blocks, naturally aligned with no explicit padding needed beyond the trailing 2 reserved
    // floats (matches this codebase's own ClusterCullMetadata/LODNodeMetadata field-ordering
    // convention: every vec3 is immediately followed by the scalar that fills its std430
    // base-alignment gap).
    struct MaterialParameters {
        maths::vec3 baseColor;
        float roughness;
        maths::vec3 emissive;
        float metallic;
        // 1.0 = fully opaque (shaded by the opaque Nanite VisBuffer pipeline, as before). < 1.0 =
        // translucent/transparent -- such a material's entity carries core::EntityFlags::
        // IsTransparent (see VulkanContext::BuildEntityData) and is routed to a separate
        // TransparentForwardPass instead: a Visibility Buffer stores exactly one winning surface
        // per pixel, so it fundamentally cannot represent "an opaque surface behind a translucent
        // one" -- matching real UE5.8, where Nanite only ever renders opaque/masked geometry and
        // translucent materials always go through a distinct forward renderer.
        float alpha;
        float _pad0 = 0.0f; // Reserved (e.g. a future IOR/refraction parameter for glass-like transparents).
        float _pad1 = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(MaterialParameters) == 48,
        "MaterialParameters must match MaterialParams in material_params.glsl exactly (std430 layout)");

    // Bounds both this CPU-side array and the matching GPU SSBO (ClusterResolvePass allocates
    // exactly kMaxMaterials slots) -- generous headroom over VulkanContext::kEntityCount (each demo
    // entity gets its own unique materialID today, see GenerateRandomMaterialTable's own comment).
    inline constexpr uint32_t kMaxMaterials = 32u;

    // One generated table: the PBR parameters themselves, plus a parallel convenience flag so
    // callers (VulkanContext::BuildEntityData, deciding each entity's core::EntityFlags::
    // IsTransparent bit) don't need to re-derive "alpha < 1.0" themselves.
    struct MaterialTable {
        std::array<MaterialParameters, kMaxMaterials> params{};
        std::array<bool, kMaxMaterials> isTransparent{};
    };

    // Randomly generates `usedSlotCount` visually-distinct PBR materials spanning UE5.8's usual
    // material categories -- dielectric, metal, emissive, translucent, transparent -- one unique
    // material per demo entity (materialID == entity index), replacing the earlier hand-authored
    // 6-recipe cycling table now that every entity can have its own look.
    //
    // Uses a FIXED seed (never std::random_device) deliberately: a demoscene "demo" is a fixed
    // procedural performance that must look identical every playback, the same reason this
    // codebase's other procedural generators (e.g. geometry::ClusterPartitioner) are seeded rather
    // than truly random. Call this with the same (usedSlotCount, seed) pair from every call site
    // that needs the table (a pure function of its two arguments) rather than threading the result
    // around if that is ever more convenient than passing it through
    // ClusterRenderPipelineCreateInfo::materialTable.
    //
    // Slots >= usedSlotCount keep a neutral matte-gray opaque default -- the same fallback every
    // out-of-range materialID already clamps to (see ClusterResolve.comp's MATERIAL_TABLE_SIZE
    // clamp).
    inline MaterialTable GenerateRandomMaterialTable(uint32_t usedSlotCount, uint32_t seed) {
        MaterialTable table{};
        for (MaterialParameters& p : table.params) {
            p = MaterialParameters{ maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);

        // HSV -> RGB (standard 6-sector formula). Procedural color generation (CLAUDE.md's own
        // "procedural colors/textures" mandate) needs to keep saturation/value under control
        // while still randomizing hue freely -- sampling R/G/B independently biases hard toward
        // muddy, desaturated colors instead of the vivid, distinct-per-primitive palette a demo
        // scene wants.
        auto hsvToRgb = [](float h, float s, float v) -> maths::vec3 {
            float c = v * s;
            float hp = h * 6.0f;
            float x = c * (1.0f - std::abs(std::fmod(hp, 2.0f) - 1.0f));
            float r1 = 0.0f, g1 = 0.0f, b1 = 0.0f;
            if (hp < 1.0f) { r1 = c; g1 = x; b1 = 0.0f; }
            else if (hp < 2.0f) { r1 = x; g1 = c; b1 = 0.0f; }
            else if (hp < 3.0f) { r1 = 0.0f; g1 = c; b1 = x; }
            else if (hp < 4.0f) { r1 = 0.0f; g1 = x; b1 = c; }
            else if (hp < 5.0f) { r1 = x; g1 = 0.0f; b1 = c; }
            else { r1 = c; g1 = 0.0f; b1 = x; }
            float m = v - c;
            return maths::vec3(r1 + m, g1 + m, b1 + m);
        };

        uint32_t clampedCount = (usedSlotCount < kMaxMaterials) ? usedSlotCount : kMaxMaterials;
        for (uint32_t i = 0; i < clampedCount; ++i) {
            MaterialParameters& p = table.params[i];
            float categoryRoll = unit(rng);
            float hue = unit(rng);

            if (categoryRoll < 0.45f) {
                // Dielectric: plain opaque non-metal, the most common category.
                p.baseColor = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.35f + 0.55f * unit(rng));
                p.roughness = 0.30f + 0.60f * unit(rng);
                p.metallic = 0.0f;
                p.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 1.0f;
            } else if (categoryRoll < 0.60f) {
                // Metal: low saturation (real metals are near-neutral/tinted, not rainbow-vivid),
                // roughness biased low for a believable polished-to-brushed look.
                p.baseColor = hsvToRgb(hue, 0.40f * unit(rng), 0.50f + 0.45f * unit(rng));
                p.roughness = 0.05f + 0.40f * unit(rng);
                p.metallic = 1.0f;
                p.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 1.0f;
            } else if (categoryRoll < 0.75f) {
                // Emissive: a self-lit dielectric base plus a bright glow color (independently
                // hued from the base color, matching e.g. a glowing crystal's own tint).
                p.baseColor = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.35f + 0.55f * unit(rng));
                p.roughness = 0.30f + 0.60f * unit(rng);
                p.metallic = 0.0f;
                p.emissive = hsvToRgb(unit(rng), 0.70f + 0.30f * unit(rng), 1.5f + 1.5f * unit(rng));
                p.alpha = 1.0f;
            } else if (categoryRoll < 0.90f) {
                // Translucent: soft semi-transparency (frosted glass / wax / leaf-like), routed to
                // TransparentForwardPass (see MaterialParameters::alpha's own comment).
                p.baseColor = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.45f + 0.50f * unit(rng));
                p.roughness = 0.20f + 0.40f * unit(rng);
                p.metallic = 0.0f;
                p.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 0.35f + 0.35f * unit(rng); // [0.35, 0.70]
            } else {
                // Transparent: clear/glass-like -- low roughness, low alpha.
                p.baseColor = hsvToRgb(hue, 0.20f + 0.40f * unit(rng), 0.55f + 0.45f * unit(rng));
                p.roughness = 0.15f * unit(rng); // [0.0, 0.15]
                p.metallic = 0.0f;
                p.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 0.05f + 0.25f * unit(rng); // [0.05, 0.30]
            }

            table.isTransparent[i] = (p.alpha < 1.0f);
        }

        return table;
    }

}
