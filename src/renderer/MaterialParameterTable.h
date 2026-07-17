#pragma once
// Small, constexpr-sized lookup keyed by geometry::ClusterIndexEntry::materialID (itself a verbatim
// copy of core::EntityData::materialID, carried through the whole LOD/cull pipeline -- see
// ClusterCullingPass.h's ClusterCullMetadata::materialID and ClusterLODSelectionPass.h's
// LODNodeMetadata::materialID). This is the RENDERER-side counterpart to
// geometry::EntityMaterialTable.h: that table drives cook-time-only per-cluster fields
// (maxWPOAmplitude/maskTextureIndex, baked into the .cache file); this one drives real Substrate PBR
// shading parameters consumed at runtime by ClusterResolve.comp/ClusterResolveBinned.comp/
// TransparentForward.frag/ReflectionGather.comp/MegaLightsShade.comp/SurfaceCache capture, uploaded
// once into a GPU SSBO by renderer::ClusterResolvePass::Init().
//
// Unmapped materialIDs (>= kMaxMaterials) are clamped by the consuming shaders before indexing, so
// they safely fall back to whatever this table's last slot holds.

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

#include "core/maths/Maths.h"

namespace renderer {

    // UE5.8 Substrate "Slab" BSDF -- the single atomic building block every Substrate material is
    // made of. GLSL-friendly, std430-compatible mirror of SubstrateSlab in
    // src/shaders/include/material_params.glsl -- 96 bytes, six {vec3, float}-or-{float x4}-shaped
    // 16-byte blocks, naturally aligned with no explicit padding needed beyond the trailing block's
    // 3 reserved floats (matches this codebase's own ClusterCullMetadata/LODNodeMetadata
    // field-ordering convention: every vec3 is immediately followed by the scalar that fills its
    // std430 base-alignment gap).
    //
    // Unlike a plain metallic-workflow material, Substrate stores specular reflectance directly as
    // F0/F90 rather than deriving it from a metallic scalar at shading time -- exactly how Substrate's
    // own Slab node works internally (its "Parameterization" dropdown, incl. a Metallic mode, all
    // resolve to Diffuse Albedo + F0 + F90 before the node ever reaches the GPU). This codebase keeps
    // a `metallic` convenience parameter only at PROCEDURAL GENERATION time (see
    // GenerateRandomMaterialTable's DeriveF0FromMetallic below) -- it is not part of the runtime
    // struct, matching real Substrate.
    struct SubstrateSlab {
        maths::vec3 diffuseAlbedo;
        float roughness;
        maths::vec3 f0;
        // Grazing-angle (90 degree incidence) Fresnel reflectance multiplier on top of F0's white
        // Schlick falloff -- Substrate's F90 edge-tint parameter. 1.0 = standard Schlick (default for
        // every category below); only deviates for special-look metals, which this table does not
        // currently generate.
        float f90Luminance = 1.0f;
        maths::vec3 emissive;
        // Substrate's Anisotropy parameter, [-1, 1]: 0 = isotropic GGX (the default), positive/
        // negative stretches the specular lobe along the procedural tangent basis (see
        // substrate_bsdf.glsl's BuildProceduralTangentBasis -- this codebase has no authored
        // per-vertex tangents, so the tangent direction is derived from the shading normal alone).
        float anisotropy = 0.0f;
        // Substrate's second specular lobe ("Haze" -- e.g. car-paint flake, brushed/polished dual
        // roughness). secondRoughnessWeight == 0.0 (default) disables it entirely at zero extra cost.
        float secondRoughness = 0.0f;
        float secondRoughnessWeight = 0.0f;
        // Substrate's Subsurface approximation: analytic wrap-diffuse (no screen-space diffusion
        // pass -- see this feature's own scope note in the Substrate integration plan). sssAmount is
        // the [0,1] blend toward wrap lighting; sssRadius is the wrap angle/falloff control.
        // sssAmount == 0.0 (default) disables it entirely.
        float sssAmount = 0.0f;
        float sssRadius = 0.0f;
        // Substrate's Fuzz parameter (cloth/velvet sheen, additive grazing term). fuzzAmount == 0.0
        // (default) disables it entirely.
        maths::vec3 fuzzColor;
        float fuzzAmount = 0.0f;
        float fuzzRoughness = 0.3f;
        float _pad0 = 0.0f;
        float _pad1 = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(SubstrateSlab) == 96,
        "SubstrateSlab must match the SubstrateSlab struct in material_params.glsl exactly (std430 layout)");

    // GLSL-friendly, std430-compatible mirror of MaterialParams in
    // src/shaders/include/material_params.glsl -- two SubstrateSlab blocks (base + optional top)
    // plus a trailing 16-byte block, matching Substrate's own "vertical layering" model: a material
    // is either a single Slab (topWeight == 0.0, the exact behavior every material had before this
    // struct existed) or a Base slab with a Top slab (Coat/Fuzz-emphasis/SSS-emphasis) layered over
    // it, composited by substrate_bsdf.glsl's EvaluateSubstrateMaterial (see that function's own
    // comment for the vertical-layer energy-conservation formula).
    struct MaterialParameters {
        SubstrateSlab base;
        SubstrateSlab top;
        // Substrate's vertical-layer Coverage/weight, [0, 1]. 0.0 (default) = fast path, `top` is
        // never evaluated -- identical cost and result to a single flat Slab material.
        float topWeight = 0.0f;
        // 1.0 = fully opaque (shaded by the opaque Nanite VisBuffer pipeline, as before). < 1.0 =
        // translucent/transparent -- such a material's entity carries core::EntityFlags::
        // IsTransparent (see VulkanContext::BuildEntityData) and is routed to a separate
        // TransparentForwardPass instead: a Visibility Buffer stores exactly one winning surface
        // per pixel, so it fundamentally cannot represent "an opaque surface behind a translucent
        // one" -- matching real UE5.8, where Nanite only ever renders opaque/masked geometry and
        // translucent materials always go through a distinct forward renderer.
        float alpha = 1.0f;
        // UE5.8 Lumen "Output Reflections" equivalent: per-material opt-in for TransparentForward.
        // frag's traced front-layer specular reflection (HWRT/SWRT, see that shader's own comment)
        // -- NOT applied to every transparent material, only ones that need it (glass/water-like),
        // matching real UE5.8 rather than paying the trace cost for every alpha-blended surface.
        // 0.0 = off (default, matches every other category incl. "Translucent" below), 1.0 = on.
        float hasReflections = 0.0f;
        float _pad0 = 0.0f;
    };
    static_assert(sizeof(MaterialParameters) == 208,
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

    // Substrate's own Slab-node "Metallic" parameterization mode, reproduced here purely as a
    // PROCEDURAL GENERATION convenience (see SubstrateSlab's own class comment for why the runtime
    // struct stores F0 directly instead) -- the standard metallic-workflow split: a dielectric's F0
    // is a flat ~4% baseline, a metal's F0 IS its albedo (no diffuse response, see
    // ClusterResolve.comp's pre-Substrate energy-conservation comment for the same fact restated at
    // the shading end).
    inline maths::vec3 DeriveF0FromMetallic(const maths::vec3& baseColor, float metallic) {
        maths::vec3 dielectricF0(0.04f, 0.04f, 0.04f);
        return maths::vec3(
            dielectricF0.x + (baseColor.x - dielectricF0.x) * metallic,
            dielectricF0.y + (baseColor.y - dielectricF0.y) * metallic,
            dielectricF0.z + (baseColor.z - dielectricF0.z) * metallic);
    }

    // Randomly generates `usedSlotCount` visually-distinct Substrate materials spanning UE5.8's usual
    // Slab categories -- dielectric, metal, emissive, translucent, transparent -- plus, layered on
    // top of those five, a chance of a Substrate vertical-layer Top slab (Clear Coat or Fuzz/Cloth),
    // anisotropy on metals, a second specular lobe on metals (car-paint flake), and subsurface on
    // dielectrics (organic/foliage/wax) -- one unique material per demo entity (materialID == entity
    // index), replacing the earlier hand-authored 6-recipe cycling table now that every entity can
    // have its own look.
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
            p = MaterialParameters{};
            p.base.diffuseAlbedo = maths::vec3(0.6f, 0.6f, 0.6f);
            p.base.roughness = 0.8f;
            p.base.f0 = DeriveF0FromMetallic(p.base.diffuseAlbedo, 0.0f);
            p.alpha = 1.0f;
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
            float metallic = 0.0f;

            if (categoryRoll < 0.45f) {
                // Dielectric: plain opaque non-metal, the most common category.
                p.base.diffuseAlbedo = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.35f + 0.55f * unit(rng));
                p.base.roughness = 0.30f + 0.60f * unit(rng);
                metallic = 0.0f;
                p.base.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 1.0f;

                // Substrate Subsurface (organic/foliage/wax look, CLAUDE.md's own procedural-foliage
                // goal): a slice of dielectrics get analytic wrap-diffuse scattering.
                if (unit(rng) < 0.25f) {
                    p.base.sssAmount = 0.35f + 0.55f * unit(rng);
                    p.base.sssRadius = 0.25f + 0.50f * unit(rng);
                }
            } else if (categoryRoll < 0.60f) {
                // Metal: low saturation (real metals are near-neutral/tinted, not rainbow-vivid),
                // roughness biased low for a believable polished-to-brushed look.
                p.base.diffuseAlbedo = hsvToRgb(hue, 0.40f * unit(rng), 0.50f + 0.45f * unit(rng));
                p.base.roughness = 0.05f + 0.40f * unit(rng);
                metallic = 1.0f;
                p.base.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 1.0f;

                // Substrate Anisotropy (brushed-metal look).
                if (unit(rng) < 0.35f) {
                    p.base.anisotropy = (unit(rng) * 2.0f - 1.0f) * (0.4f + 0.5f * unit(rng));
                }
                // Substrate second specular lobe / Haze (car-paint flake, polished-with-microflake).
                if (unit(rng) < 0.20f) {
                    p.base.secondRoughness = 0.02f + 0.08f * unit(rng);
                    p.base.secondRoughnessWeight = 0.30f + 0.40f * unit(rng);
                }
            } else if (categoryRoll < 0.75f) {
                // Emissive: a self-lit dielectric base plus a bright glow color (independently
                // hued from the base color, matching e.g. a glowing crystal's own tint).
                p.base.diffuseAlbedo = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.35f + 0.55f * unit(rng));
                p.base.roughness = 0.30f + 0.60f * unit(rng);
                metallic = 0.0f;
                p.base.emissive = hsvToRgb(unit(rng), 0.70f + 0.30f * unit(rng), 1.5f + 1.5f * unit(rng));
                p.alpha = 1.0f;
            } else if (categoryRoll < 0.90f) {
                // Translucent: soft semi-transparency (frosted glass / wax / leaf-like), routed to
                // TransparentForwardPass (see MaterialParameters::alpha's own comment).
                p.base.diffuseAlbedo = hsvToRgb(hue, 0.35f + 0.50f * unit(rng), 0.45f + 0.50f * unit(rng));
                p.base.roughness = 0.20f + 0.40f * unit(rng);
                metallic = 0.0f;
                p.base.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 0.35f + 0.35f * unit(rng); // [0.35, 0.70]
                p.hasReflections = 0.0f; // Frosted/soft translucency -- no clear image to reflect.
            } else {
                // Transparent: clear/glass-like -- low roughness, low alpha, traced reflections on
                // (see MaterialParameters::hasReflections's own comment).
                p.base.diffuseAlbedo = hsvToRgb(hue, 0.20f + 0.40f * unit(rng), 0.55f + 0.45f * unit(rng));
                p.base.roughness = 0.15f * unit(rng); // [0.0, 0.15]
                metallic = 0.0f;
                p.base.emissive = maths::vec3(0.0f, 0.0f, 0.0f);
                p.alpha = 0.05f + 0.25f * unit(rng); // [0.05, 0.30]
                p.hasReflections = 1.0f;
            }

            p.base.f0 = DeriveF0FromMetallic(p.base.diffuseAlbedo, metallic);
            // Real Substrate Slabs have no "metallic" concept at shading time -- a metal's diffuse
            // albedo IS zero (all incident light either reflects specularly via F0 above or is
            // absorbed), so a metal's hue lives entirely in F0, not diffuseAlbedo. This replaces the
            // old runtime "diffuseAlbedo*(1-metallic)" energy-conservation multiply (there is no
            // metallic scalar left to multiply by at shading time) with the physically-correct
            // authoring-time equivalent: bake the zero directly into the Slab.
            if (metallic > 0.5f) {
                p.base.diffuseAlbedo = maths::vec3(0.0f, 0.0f, 0.0f);
            }

            // Substrate vertical layering: a Top slab (Clear Coat or Fuzz/Cloth) over the Base slab
            // just rolled above -- independent of category, matching real Substrate (any Slab can be
            // coated or fuzzed). Skipped for the Transparent category: a glass Top coat over a glass
            // Base is not a look this generator produces, and TransparentForward.frag's own
            // hasReflections trace already covers that category's "clear coat" look.
            if (categoryRoll < 0.90f) {
                float topRoll = unit(rng);
                if (topRoll < 0.15f) {
                    // Clear Coat: thin dielectric coat (wet rock / lacquered wood), near-zero
                    // roughness, always dielectric F0 regardless of the base's own metallic-ness.
                    p.topWeight = 0.55f + 0.40f * unit(rng);
                    p.top.diffuseAlbedo = maths::vec3(0.0f, 0.0f, 0.0f); // Coat itself has no diffuse response.
                    p.top.roughness = 0.02f + 0.10f * unit(rng);
                    p.top.f0 = maths::vec3(0.04f, 0.04f, 0.04f);
                } else if (topRoll < 0.28f) {
                    // Fuzz/Cloth: moss/velvet sheen layered over the base color.
                    p.topWeight = 0.40f + 0.45f * unit(rng);
                    p.top.diffuseAlbedo = p.base.diffuseAlbedo;
                    p.top.roughness = 0.6f + 0.3f * unit(rng);
                    p.top.f0 = maths::vec3(0.04f, 0.04f, 0.04f);
                    p.top.fuzzColor = hsvToRgb(hue, 0.30f + 0.40f * unit(rng), 0.55f + 0.35f * unit(rng));
                    p.top.fuzzAmount = 0.5f + 0.4f * unit(rng);
                    p.top.fuzzRoughness = 0.4f + 0.4f * unit(rng);
                }
            }

            table.isTransparent[i] = (p.alpha < 1.0f);
        }

        return table;
    }

}
