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
#include <cstdint>

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
    // GenerateShowcaseMaterialTable's DeriveF0FromMetallic below) -- it is not part of the runtime
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
        // Screen-space Subsurface Scattering (UE5.8 rendering-parity gap G4, "Subsurface Profile"
        // shading model): the world-space diffusion radius fed to renderer::SubsurfaceScatteringPass'
        // separable screen-space blur (the Jimenez/Burley "Separable SSS" technique). 0.0 (default) =
        // disabled (the pixel is never touched by that pass -- exactly the pre-G4 behavior, at zero
        // extra cost). DISTINCT from sssAmount/sssRadius above: those drive the cheap analytic
        // wrap-diffuse baked directly into EvaluateSlabDiffuse (substrate_bsdf.glsl), whereas this
        // drives a real post-lighting separable screen-space diffusion pass -- exactly Substrate's
        // own distinction between its "Subsurface" (wrap) and "Subsurface Profile" (screen-space
        // diffusion) shading models. A material authors ONE or the other, not both (setting both
        // would double-count subsurface transport -- see GenerateShowcaseMaterialTable's own waxy SSS
        // showcase recipe, which sets this and leaves sssAmount at 0). Occupies what was _pad0.
        float sssProfileScale = 0.0f;
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
        // Wave 2 (UE5.8 Substrate iridescence layer): thin-film interference "shimmer" -- see
        // renderer::IridescenceMaterialParams' own comment (MaterialParameterTable.h intentionally
        // keeps these as two flat scalars here rather than embedding that struct directly, matching
        // every other field in this table). 0.0 = off (default, zero extra cost -- see
        // substrate_bsdf.glsl's EvaluateIridescence early-exit). iridescenceThickness in [0,1] maps
        // to a [10,500]nm film thickness (see EvaluateIridescence's own comment).
        float iridescenceAmount = 0.0f;
        float iridescenceThickness = 0.5f;
        float _padIridescence0 = 0.0f;
        float _padIridescence1 = 0.0f;
    };
    static_assert(sizeof(MaterialParameters) == 224,
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
    // mistakenly reused that exact slot and silently clobbered it).
    // Assigned explicitly to exactly one entity (the Icosphere, VulkanContext::kHeroEntityIndex) in
    // VulkanContext::BuildEntityData(), never reached via the per-entity assignment loop.
    inline constexpr uint32_t kHeroMaterialID = 15u;

    // Phase 7b (UE5.8 parity roadmap, terrain heightfield): the procedural terrain entity's
    // fallback material. Reserved one past kHeroMaterialID, same "never collides with a real
    // entity's showcase material" convention. Unlike glass/hero, this recipe's baseColor is only a
    // FALLBACK: ClusterResolve.comp/ClusterResolveBinned.comp override it per-pixel with a
    // height/slope biome blend (grass/rock/snow, see terrain_shading.glsl) for any cluster tagged
    // with this materialID -- the terrain still renders through the normal opaque Nanite path (no
    // ClusterLODCompact.comp exclusion needed, unlike the hero entity: terrain is fully opaque and
    // has no runtime-displaced/tessellated representation to protect).
    inline constexpr uint32_t kTerrainMaterialID = 16u;

    // Phase 7c (UE5.8 parity roadmap, water/erosion): the water plane's material. Reserved one
    // past kTerrainMaterialID, same "never collides with a real entity's showcase material"
    // convention -- assigned explicitly to exactly one entity (the water plane, see
    // VulkanContext::kWaterEntityIndex). Rendered ONLY by renderer::WaterForwardPass; never
    // reaches the opaque Nanite resolve shaders (its entity is unconditionally excluded from that
    // path via core::EntityFlags::IsTransparent, same mechanism as the hero entity -- see
    // VulkanContext::BuildEntityData()'s own comment). `alpha` here is repurposed as the MAXIMUM
    // depth-based absorption blend strength (not a fixed-function blend alpha -- WaterForward.frag
    // composes manually against a background snapshot, see that pass's own header comment).
    inline constexpr uint32_t kWaterMaterialID = 17u;

    // Runtime World Partition streaming pool (VulkanContext::kStreamingUnitCount): one opaque
    // material per archetype shape (Rock/Bush/Tree/Debris, see VulkanContext::
    // kStreamingArchetypeShapeCount), reserved one past kWaterMaterialID, same "never collides with
    // a real entity's showcase material" convention. Both the coarse and fine mesh variant of a
    // streaming unit share their shape's single material -- only the geometry detail differs
    // between them, not the look.
    inline constexpr uint32_t kStreamingArchetypeMaterialIDBase = 18u; // Occupies 18..21 (4 shapes).

    // Procedural tree generator (renderer::ProceduralTreePass, see CLAUDE.md's "Arbres (generes
    // par du code style speedtree)" requirement): two reserved slots, one past every streaming
    // archetype above, same "never collides with a real entity's showcase material" convention.
    // Split into two IDs (not one) because this codebase's cluster/material pipeline assigns
    // exactly one materialID per whole entity mesh (see this file's own class comment) -- a tree's
    // bark and its foliage need visibly different looks (and, for the foliage, an opacity-cutout
    // mask + WPO wind sway neither the bark nor any other existing entity needs, see
    // geometry::EntityMaterialTable.h's matching case for kTreeLeafMaterialID), so they are baked
    // as two separate co-located entities, not one entity with per-vertex-varying materials.
    inline constexpr uint32_t kTreeBarkMaterialID = 22u;
    inline constexpr uint32_t kTreeLeafMaterialID = 23u;

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

    // Builds one Substrate Slab from the old flat metallic-workflow parameterization
    // (baseColor/roughness/emissive/metallic) every recipe below is still most naturally authored
    // in -- a metal's diffuseAlbedo is zeroed (see SubstrateSlab's own class comment: a metal's hue
    // lives entirely in F0, never in diffuseAlbedo, matching how a real Substrate Slab is authored).
    inline SubstrateSlab MakeBaseSlab(const maths::vec3& baseColor, float roughness, const maths::vec3& emissive, float metallic) {
        SubstrateSlab slab{};
        slab.diffuseAlbedo = (metallic > 0.5f) ? maths::vec3(0.0f, 0.0f, 0.0f) : baseColor;
        slab.roughness = roughness;
        slab.f0 = DeriveF0FromMetallic(baseColor, metallic);
        slab.emissive = emissive;
        return slab;
    }

    // Hand-authors one explicit, named PBR material per base-scene entity (materialID == entity
    // index) so the scene reads as a deliberate feature gallery -- each entity demonstrates exactly
    // one named category -- instead of a randomly-rolled look. Mirrors VulkanContext::GridSlot's
    // own zone layout 1:1:
    //
    //   0  Box        -> METAL A "chrome"      7  Capsule    -> METAL B "gold" (anisotropic)
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
    // Substrate integration: two recipes additionally demonstrate the vertical-layering Top slab
    // (see substrate_bsdf.glsl's own EvaluateSubstrateMaterial) on top of their existing base look
    // -- Metal B "gold" (slot 7) gains Anisotropy for a brushed-metal look, matching this function's
    // own pre-existing "brushed look" comment on that recipe; the hero tessellated rock
    // (kHeroMaterialID) gains a thin Clear Coat, matching this codebase's own pre-existing "a
    // subtle sheen" comment on that recipe. Every other slot keeps topWeight == 0.0 (no Top slab),
    // preserving this table's curated per-entity look exactly.
    //
    // Slots beyond the entities explicitly listed below (indices 15..kMaxMaterials-1) keep a
    // neutral matte-gray opaque default -- the same fallback every out-of-range materialID already
    // clamps to (see ClusterResolve.comp's MATERIAL_TABLE_SIZE clamp).
    inline MaterialTable GenerateShowcaseMaterialTable() {
        MaterialTable table{};
        for (MaterialParameters& p : table.params) {
            p = MaterialParameters{};
            p.base = MakeBaseSlab(maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        }

        // Metal A "chrome" (Box, slot 0): near-white, low roughness, fully metallic.
        table.params[0].base = MakeBaseSlab(maths::vec3(0.85f, 0.85f, 0.88f), 0.08f, maths::vec3(0.0f, 0.0f, 0.0f), 1.0f);

        // WPO/displacement (Cone, slot 1): vivid teal dielectric so the WPO sway
        // (geometry::EntityMaterialTable's materialID==1 case) reads clearly against a plain color.
        table.params[1].base = MakeBaseSlab(maths::vec3(0.15f, 0.65f, 0.55f), 0.50f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Nanite A (Icosphere, slot 2): neutral gray -- keeps cluster/LOD debug views (Numpad '*')
        // free of color noise.
        table.params[2].base = MakeBaseSlab(maths::vec3(0.65f, 0.65f, 0.68f), 0.55f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Dielectric A (Plane, slot 3): matte warm orange, fully opaque non-metal.
        table.params[3].base = MakeBaseSlab(maths::vec3(0.75f, 0.35f, 0.12f), 0.85f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Transparent/glass (Sphere, slot 4): clear pale-blue, near-zero roughness, low alpha,
        // traced front-layer reflections on (see MaterialParameters::hasReflections's own comment).
        // Wave 2: also this scene's iridescence showcase -- a thin-film soap-bubble/oil-film shimmer
        // on top of the glass look (see MaterialParameters::iridescenceAmount's own comment); a
        // moderate 0.4 amount keeps the underlying glass color legible rather than fully overwritten
        // by the rainbow tint, ~185nm film thickness (thin end of the mapped range -- reads as a
        // tighter, more colorful interference pattern than a thicker film would).
        table.params[4].base = MakeBaseSlab(maths::vec3(0.75f, 0.85f, 0.95f), 0.03f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[4].alpha = 0.12f;
        table.params[4].hasReflections = 1.0f;
        table.params[4].iridescenceAmount = 0.4f;
        table.params[4].iridescenceThickness = 0.35f;

        // Translucent (Torus, slot 5): soft frosted violet, mid alpha, no reflections (no clear
        // image to reflect through a frosted surface). Phase PP3 (post-process stack roadmap): also
        // this scene's own Heat Distortion & Refraction showcase -- see MaterialParameters::
        // heatDistortion's own comment.
        table.params[5].base = MakeBaseSlab(maths::vec3(0.55f, 0.35f, 0.75f), 0.35f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[5].alpha = 0.5f;
        table.params[5].heatDistortion = 0.7f;

        // Emissive (Tube, slot 6): near-black base, bright warm self-lit glow.
        table.params[6].base = MakeBaseSlab(maths::vec3(0.05f, 0.05f, 0.05f), 0.40f, maths::vec3(3.0f, 1.4f, 0.2f), 0.0f);

        // Metal B "gold" (Capsule, slot 7): warm-tinted, slightly higher roughness than chrome for
        // a brushed look, fully metallic -- Substrate Anisotropy makes that "brushed" look real
        // (stretches the specular lobe along the procedural tangent basis instead of just being a
        // higher-roughness isotropic blur).
        table.params[7].base = MakeBaseSlab(maths::vec3(0.9f, 0.7f, 0.25f), 0.18f, maths::vec3(0.0f, 0.0f, 0.0f), 1.0f);
        table.params[7].base.anisotropy = 0.6f;

        // MegaLights hero (Cylinder, slot 8): neutral mid-gray so the ~200 stochastic point lights'
        // own colors read clearly rather than mixing with a tinted base.
        table.params[8].base = MakeBaseSlab(maths::vec3(0.7f, 0.7f, 0.7f), 0.6f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Dielectric B (Pyramid, slot 9): matte deep green, fully opaque non-metal.
        table.params[9].base = MakeBaseSlab(maths::vec3(0.15f, 0.55f, 0.25f), 0.75f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Screen-space Subsurface Scattering showcase (TorusKnot, slot 10) -- UE5.8 rendering-parity
        // gap G4, "Subsurface Profile" shading model. Repurposes what was the "Nanite B neutral"
        // slot: a torus knot's many thin, self-shadowing tubes are the ideal shape to show light
        // visibly bleeding THROUGH thin geometry and softly wrapping past the sun terminator, exactly
        // the effect renderer::SubsurfaceScatteringPass produces (vs. the sharp Lambertian falloff
        // every other opaque material here has). It is still fully-opaque neutral Nanite geometry, so
        // it keeps serving the "Nanite B" role too -- and every per-cluster debug view (Numpad '*')
        // hash-colors it regardless of this base color, so those visualizations are unaffected.
        // A warm ivory/wax base color (not pure white) so the diffusion profile's characteristic
        // reddish edge-bleed reads clearly; moderate-high roughness (soft, low specular -- an organic
        // waxy surface, not glossy); dielectric; reflections OFF (default). sssProfileScale drives the
        // world-space diffusion radius (see renderer::SubstrateSlab::sssProfileScale); sssAmount is
        // deliberately left at 0 -- this recipe uses the real screen-space diffusion, NOT the cheap
        // analytic wrap-diffuse, and using both would double-count subsurface transport.
        table.params[10].base = MakeBaseSlab(maths::vec3(0.86f, 0.72f, 0.60f), 0.45f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[10].base.sssProfileScale = 0.30f;

        // Lumen/GI hero (ChamferBox, slot 11): near-white neutral so it best picks up the red/green
        // bounce light from the 2 corner walls, with no tint of its own to muddy the result.
        table.params[11].base = MakeBaseSlab(maths::vec3(0.85f, 0.85f, 0.85f), 0.6f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Lumen/GI corner walls (slots 12/13): saturated matte red/green -- the classic Cornell-box
        // choice, maximizing the visible color-bounce contrast on the neutral hero above.
        table.params[12].base = MakeBaseSlab(maths::vec3(0.75f, 0.12f, 0.12f), 0.9f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[13].base = MakeBaseSlab(maths::vec3(0.12f, 0.75f, 0.12f), 0.9f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Floor (slot 14): neutral matte gray ground -- explicitly always opaque (a see-through
        // 300x300m ground plane would read as broken, not stylized, and would be the worst-case
        // primitive for TransparentForwardPass's unsorted-between-entities blending, being by far
        // the largest on screen).
        table.params[14].base = MakeBaseSlab(maths::vec3(0.6f, 0.6f, 0.6f), 0.8f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        for (uint32_t i = 0; i < kMaxMaterials; ++i) {
            table.isTransparent[i] = (table.params[i].alpha < 1.0f);
        }

        // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): hero rocky/stone recipe --
        // earthy brown-gray, high roughness (no sharp specular, matching a weathered rock
        // surface), fully opaque, reflections ON (a subtle sheen, showcasing
        // renderer::TessellationPass's own GGX-VNDF reflection trace the same way the
        // "Transparent/glass" material above showcases it for TransparentForwardPass). Rendered
        // ONLY by renderer::TessellationPass; never reaches the opaque Nanite resolve shaders
        // (its entity is unconditionally excluded from that path via core::EntityFlags::
        // IsTransparent, see VulkanContext::BuildEntityData()'s own comment) nor
        // TransparentForwardPass (table.isTransparent[kHeroMaterialID] stays false).
        // kHeroMaterialID (15) sits past every hand-authored slot above (0-14, one per real
        // entity -- see this function's own zone-layout comment), so the loop above never touches
        // it -- still at this function's own top-of-array neutral default until overwritten here.
        // Substrate: a thin Clear Coat Top slab is the actual mechanism behind that "subtle sheen"
        // -- a wet-rock look (weathered stone with a light coat, e.g. after rain), rather than the
        // rock's own base roughness alone trying to fake a sheen it physically wouldn't have.
        table.params[kHeroMaterialID].base = MakeBaseSlab(maths::vec3(0.40f, 0.33f, 0.27f), 0.92f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[kHeroMaterialID].hasReflections = 1.0f;
        table.params[kHeroMaterialID].topWeight = 0.35f;
        table.params[kHeroMaterialID].top.roughness = 0.10f;
        table.params[kHeroMaterialID].top.f0 = maths::vec3(0.04f, 0.04f, 0.04f);

        // Phase 7b (UE5.8 parity roadmap, terrain heightfield): terrain fallback -- grass-green,
        // high roughness, dielectric. Overridden per-pixel by ComputeTerrainAlbedo()'s height/slope
        // blend wherever the terrain material is shaded (writes mat.base.diffuseAlbedo directly, see
        // ClusterResolve.comp/ClusterResolveBinned.comp's own Step 3 comment); this diffuseAlbedo
        // only matters where that blend degenerates (e.g. flat low ground -> grass anyway).
        // kTerrainMaterialID (16) likewise sits past every hand-authored slot, untouched by the
        // isTransparent loop above.
        table.params[kTerrainMaterialID].base = MakeBaseSlab(maths::vec3(0.24f, 0.42f, 0.15f), 0.88f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Phase 7c (UE5.8 parity roadmap, water/erosion): water -- deep blue-teal absorption tint
        // (reused directly as WaterForward.frag's depth-based Beer-Lambert tint color, not just a
        // flat diffuseAlbedo), near-mirror roughness (calm water), alpha repurposed as the maximum
        // absorption blend strength at full depth (0.85 -- some background always shows through
        // even at the deepest basin point). kWaterMaterialID (17) likewise sits past every
        // hand-authored slot, untouched by the isTransparent loop above (WaterForwardPass reads
        // this material directly via its own single-element buffer, not this SSBO array -- see
        // that pass' own class comment).
        table.params[kWaterMaterialID].base = MakeBaseSlab(maths::vec3(0.02f, 0.10f, 0.18f), 0.04f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);
        table.params[kWaterMaterialID].alpha = 0.85f;

        // Runtime World Partition streaming pool: 4 opaque archetype recipes, indices
        // kStreamingArchetypeMaterialIDBase..+3 (18..21) -- likewise past every hand-authored slot
        // above, untouched by the isTransparent loop (all stay fully opaque).
        table.params[kStreamingArchetypeMaterialIDBase + 0].base =
            MakeBaseSlab(maths::vec3(0.45f, 0.42f, 0.40f), 0.85f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f); // Rock: matte gray-brown.
        table.params[kStreamingArchetypeMaterialIDBase + 1].base =
            MakeBaseSlab(maths::vec3(0.18f, 0.45f, 0.15f), 0.75f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f); // Bush: matte green.
        table.params[kStreamingArchetypeMaterialIDBase + 2].base =
            MakeBaseSlab(maths::vec3(0.28f, 0.20f, 0.12f), 0.80f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f); // Tree: matte dark brown.
        table.params[kStreamingArchetypeMaterialIDBase + 3].base =
            MakeBaseSlab(maths::vec3(0.55f, 0.30f, 0.18f), 0.70f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f); // Debris: rusty orange-brown.

        // Procedural tree generator (renderer::ProceduralTreePass): bark -- matte dark brown, high
        // roughness, non-metal, fully opaque (no cutout mask -- geom_tree_bark.comp's cylinder mesh
        // is already solid geometry, not a cutout card). kTreeBarkMaterialID likewise sits past
        // every hand-authored slot above, untouched by the isTransparent loop.
        table.params[kTreeBarkMaterialID].base =
            MakeBaseSlab(maths::vec3(0.30f, 0.20f, 0.13f), 0.88f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        // Procedural tree generator: foliage -- matte green, high roughness, non-metal. Fully
        // OPAQUE (alpha == 1.0, isTransparent stays false): the leaf cards use the opacity-CUTOUT
        // mechanism (geometry::EntityMaterialTable.h's kTreeLeafMaterialID case, sampled via
        // mask_sampling.glsl's hard discard/soft-edge-feather in ClusterRaster/ClusterResolve),
        // which is a hard alpha-test on an otherwise fully-opaque surface -- NOT the separate alpha-
        // BLEND transparency mechanism (MaterialParameters::alpha < 1.0) TransparentForwardPass
        // handles, matching how every other masked/cutout material in a Nanite-style pipeline stays
        // on the opaque VisBuffer path rather than being routed to a forward blend pass.
        table.params[kTreeLeafMaterialID].base =
            MakeBaseSlab(maths::vec3(0.16f, 0.42f, 0.12f), 0.75f, maths::vec3(0.0f, 0.0f, 0.0f), 0.0f);

        return table;
    }

}
