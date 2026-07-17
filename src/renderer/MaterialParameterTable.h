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
#include <cstdint>

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
        // Phase PP3 (post-process stack roadmap): 0.0 = off (default), >0.0 = this material writes a
        // procedural, animated screen-space refraction offset into renderer::TransparentForwardPass's
        // own second (RG16F) color attachment -- renderer::PostProcessPass's composite shader then
        // samples that buffer and distorts the UV it reads the HDR scene through, exactly UE5.8's own
        // material-authored "Refraction" mechanism (a post-process material samples SceneColor at an
        // offset UV a translucent material's shader writes), not a single global distortion knob. See
        // TransparentForward.frag's own end-of-main() comment for the actual noise formula.
        float heatDistortion = 0.0f;
        // UE5.8 Lumen "Output Reflections" equivalent: per-material opt-in for TransparentForward.
        // frag's traced front-layer specular reflection (HWRT/SWRT, see that shader's own comment)
        // -- NOT applied to every transparent material, only ones that need it (glass/water-like),
        // matching real UE5.8 rather than paying the trace cost for every alpha-blended surface.
        // 0.0 = off (default, matches every other category incl. "Translucent" below), 1.0 = on.
        float hasReflections = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(MaterialParameters) == 48,
        "MaterialParameters must match MaterialParams in material_params.glsl exactly (std430 layout)");

    // Bounds both this CPU-side array and the matching GPU SSBO (ClusterResolvePass allocates
    // exactly kMaxMaterials slots) -- generous headroom over VulkanContext::kEntityCount (each demo
    // entity gets its own unique materialID today, see GenerateShowcaseMaterialTable's own comment).
    inline constexpr uint32_t kMaxMaterials = 32u;

    // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): the single tessellated/displaced
    // hero material. Deliberately set to VulkanContext::kEntityCount (15) -- one past the highest
    // entity-index-derived materialID GenerateShowcaseMaterialTable() ever hand-authors (entities
    // 0..14 each get materialID == their own index, see that function's own zone-layout comment)
    // -- so this reserved slot can never collide with a real entity's own showcase material (in
    // particular slot 13, the green Cornell-box wall -- an earlier revision of this constant
    // mistakenly reused that exact slot and silently clobbered it; see [[feedback_clean_merge_not_correct]]).
    // Assigned explicitly to exactly one entity (the Icosphere, VulkanContext::kHeroEntityIndex) in
    // VulkanContext::BuildEntityData(), never reached via the per-entity assignment loop.
    inline constexpr uint32_t kHeroMaterialID = 15u;

    // One generated table: the PBR parameters themselves, plus a parallel convenience flag so
    // callers (VulkanContext::BuildEntityData, deciding each entity's core::EntityFlags::
    // IsTransparent bit) don't need to re-derive "alpha < 1.0" themselves.
    struct MaterialTable {
        std::array<MaterialParameters, kMaxMaterials> params{};
        std::array<bool, kMaxMaterials> isTransparent{};
    };

    // Hand-authors one explicit, named PBR material per base-scene entity (materialID == entity
    // index) so the scene reads as a deliberate feature gallery -- each entity demonstrates exactly
    // one named category -- instead of a randomly-rolled look. Mirrors VulkanContext::GridSlot's
    // own zone layout 1:1:
    //
    //   0  Box        -> METAL A "chrome"      7  Capsule    -> METAL B "gold"
    //   1  Cone        -> WPO/displacement      8  Cylinder    -> MegaLights hero (neutral,
    //      (paired with geometry::EntityMaterialTable's materialID==1 sway case)   lets the ~200
    //   2  Icosphere   -> Nanite A (neutral)                     stochastic lights read clearly)
    //   3  Plane       -> Dielectric A          9  Pyramid     -> Dielectric B
    //   4  Sphere       -> Transparent/glass    10 TorusKnot   -> Nanite B (neutral)
    //   5  Torus        -> Translucent          11 ChamferBox  -> Lumen/GI hero (neutral, picks up
    //   6  Tube         -> Emissive                                 bounce color from the 2 walls)
    //   12 Wall A (red), 13 Wall B (green)  -> Lumen/GI corner walls
    //   14 Floor                              -> neutral opaque ground
    //
    // Fully deterministic (no RNG at all): a demoscene "demo" is a fixed procedural performance
    // that must look identical every playback, the same reason this codebase's other procedural
    // generators (e.g. geometry::ClusterPartitioner) are seeded rather than truly random -- here
    // taken to its logical conclusion since every entity's role is fixed by design, not sampled.
    //
    // Slots beyond the entities explicitly listed below (indices 15..kMaxMaterials-1) keep a
    // neutral matte-gray opaque default -- the same fallback every out-of-range materialID already
    // clamps to (see ClusterResolve.comp's MATERIAL_TABLE_SIZE clamp).
    inline MaterialTable GenerateShowcaseMaterialTable() {
        MaterialTable table{};
        for (MaterialParameters& p : table.params) {
            p = MaterialParameters{ maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
        }

        // Metal A "chrome" (Box, slot 0): near-white, low roughness, fully metallic.
        table.params[0] = MaterialParameters{ maths::vec3(0.85f, 0.85f, 0.88f), 0.08f, maths::vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // WPO/displacement (Cone, slot 1): vivid teal dielectric so the WPO sway
        // (geometry::EntityMaterialTable's materialID==1 case) reads clearly against a plain color.
        table.params[1] = MaterialParameters{ maths::vec3(0.15f, 0.65f, 0.55f), 0.50f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Nanite A (Icosphere, slot 2): neutral gray -- keeps cluster/LOD debug views (Numpad '*')
        // free of color noise.
        table.params[2] = MaterialParameters{ maths::vec3(0.65f, 0.65f, 0.68f), 0.55f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Dielectric A (Plane, slot 3): matte warm orange, fully opaque non-metal.
        table.params[3] = MaterialParameters{ maths::vec3(0.75f, 0.35f, 0.12f), 0.85f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Transparent/glass (Sphere, slot 4): clear pale-blue, near-zero roughness, low alpha,
        // traced front-layer reflections on (see MaterialParameters::hasReflections's own comment).
        table.params[4] = MaterialParameters{ maths::vec3(0.75f, 0.85f, 0.95f), 0.03f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 0.12f, 0.0f, 1.0f, 0.0f };

        // Translucent (Torus, slot 5): soft frosted violet, mid alpha, no reflections (no clear
        // image to reflect through a frosted surface). Phase PP3 (post-process stack roadmap): also
        // this scene's own Heat Distortion & Refraction showcase -- see MaterialParameters::
        // heatDistortion's own comment.
        table.params[5] = MaterialParameters{ maths::vec3(0.55f, 0.35f, 0.75f), 0.35f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 0.5f, 0.7f, 0.0f, 0.0f };

        // Emissive (Tube, slot 6): near-black base, bright warm self-lit glow.
        table.params[6] = MaterialParameters{ maths::vec3(0.05f, 0.05f, 0.05f), 0.40f, maths::vec3(3.0f, 1.4f, 0.2f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Metal B "gold" (Capsule, slot 7): warm-tinted, slightly higher roughness than chrome for
        // a brushed look, fully metallic.
        table.params[7] = MaterialParameters{ maths::vec3(0.9f, 0.7f, 0.25f), 0.18f, maths::vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // MegaLights hero (Cylinder, slot 8): neutral mid-gray so the ~200 stochastic point lights'
        // own colors read clearly rather than mixing with a tinted base.
        table.params[8] = MaterialParameters{ maths::vec3(0.7f, 0.7f, 0.7f), 0.6f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Dielectric B (Pyramid, slot 9): matte deep green, fully opaque non-metal.
        table.params[9] = MaterialParameters{ maths::vec3(0.15f, 0.55f, 0.25f), 0.75f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Nanite B (TorusKnot, slot 10): neutral gray, distinct tint from slot 2 but same intent.
        table.params[10] = MaterialParameters{ maths::vec3(0.6f, 0.6f, 0.7f), 0.5f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Lumen/GI hero (ChamferBox, slot 11): near-white neutral so it best picks up the red/green
        // bounce light from the 2 corner walls, with no tint of its own to muddy the result.
        table.params[11] = MaterialParameters{ maths::vec3(0.85f, 0.85f, 0.85f), 0.6f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Lumen/GI corner walls (slots 12/13): saturated matte red/green -- the classic Cornell-box
        // choice, maximizing the visible color-bounce contrast on the neutral hero above.
        table.params[12] = MaterialParameters{ maths::vec3(0.75f, 0.12f, 0.12f), 0.9f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
        table.params[13] = MaterialParameters{ maths::vec3(0.12f, 0.75f, 0.12f), 0.9f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        // Floor (slot 14): neutral matte gray ground -- explicitly always opaque (a see-through
        // 300x300m ground plane would read as broken, not stylized, and would be the worst-case
        // primitive for TransparentForwardPass's unsorted-between-entities blending, being by far
        // the largest on screen).
        table.params[14] = MaterialParameters{ maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

        for (uint32_t i = 0; i < kMaxMaterials; ++i) {
            table.isTransparent[i] = (table.params[i].alpha < 1.0f);
        }

        // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): hero rocky/stone recipe --
        // earthy brown-gray, high roughness (no sharp specular, matching a weathered rock
        // surface), fully opaque, reflections ON (a subtle sheen, showcasing
        // renderer::HeroTessellationPass's own GGX-VNDF reflection trace the same way the
        // "Transparent/glass" material above showcases it for TransparentForwardPass). Rendered
        // ONLY by renderer::HeroTessellationPass; never reaches the opaque Nanite resolve shaders
        // (its entity is unconditionally excluded from that path via core::EntityFlags::
        // IsTransparent, see VulkanContext::BuildEntityData()'s own comment) nor
        // TransparentForwardPass (table.isTransparent[kHeroMaterialID] stays false).
        // kHeroMaterialID (15) sits past every hand-authored slot above (0-14, one per real
        // entity -- see this function's own zone-layout comment), so the loop above never touches
        // it -- still at this function's own top-of-array neutral default until overwritten here.
        table.params[kHeroMaterialID] = MaterialParameters{
            maths::vec3(0.40f, 0.33f, 0.27f), 0.92f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };

        return table;
    }

}
