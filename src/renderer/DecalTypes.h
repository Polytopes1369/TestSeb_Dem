#pragma once
// UE5.8-parity gap G1 (Decal system): CPU-side authoring + GPU-upload types for the deferred,
// fully-procedural projected-decal system driven by renderer::DecalProjectionPass / DecalProject.comp.
//
// A decal is an oriented BOX volume in world space (center + an orthonormal right/up basis + per-axis
// half-extents). Every visible surface fragment whose reconstructed world position falls inside that
// box receives the decal's procedural material patch, projected along the box's third (right x up)
// axis -- exactly how a real UE5.8 deferred decal projects a material patch onto whatever geometry
// sits inside its decal frustum/box. There are NO texture assets anywhere: every decal "look" (moss,
// rust, cracks, paint markings) is generated purely from the noise functions already in this
// codebase's shader includes (displacement_noise.glsl), satisfying CLAUDE.md's hard "aucun data dans
// mon .exe" rule the same way every other visual in this engine is procedural.
//
// This is the RENDERER-side authoring counterpart to geometry::DecalBVH (the spatial structure the
// GPU traverses): GenerateShowcaseDecals() below produces both the GPU instance array AND the parallel
// world-space AABBs geometry::BuildDecalBVH consumes, so the two always describe the same decal set.
// It mirrors renderer::GenerateShowcaseMaterialTable's own "one deterministic hand-authored showcase
// set, no RNG" convention (a demoscene must look identical every playback).

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "geometry/DecalBVH.h"

namespace renderer {

    // Procedural decal "material" kind -- selects which noise-driven pattern DecalProject.comp's
    // EvaluateProceduralDecal() generates. Kept in lockstep with decal_common.glsl's own
    // DECAL_TYPE_* constants (there is no shared enum across the C++/GLSL boundary, same per-shader-
    // mirror convention this codebase already uses for e.g. the material params).
    enum class DecalType : uint32_t {
        Moss = 0u,   // Mottled green/algae organic patch with soft, noise-eroded edges.
        Rust = 1u,   // Orange-brown oxidation stain with vertical drip streaks.
        Cracks = 2u, // Dark thin fracture lines carved into the surface (also perturbs the normal).
        Paint = 3u,  // Hard-edged saturated painted hazard stripes/marking (optionally emissive).
    };

    // std430/std140-compatible GPU mirror -- 144 bytes, nine 16-byte blocks (a mat4 plus five vec4s),
    // naturally aligned with no explicit padding needed. Uploaded once into renderer::
    // DecalProjectionPass's decal SSBO. Field packing follows this codebase's "every vec3 is followed
    // by the scalar that fills its 16-byte slot" convention (see renderer::SubstrateSlab).
    struct alignas(16) DecalInstanceGPU {
        // World -> decal-local unit-cube transform. A world position p is INSIDE this decal iff every
        // component of (worldToDecal * vec4(p, 1)).xyz lies in [-1, 1]. Built as the inverse of the
        // decal's local-to-world box transform (see MakeDecalInstance below).
        maths::mat4 worldToDecal;
        maths::vec4 axisRightWorld{}; // xyz = unit world-space box "right" axis (decal-local +X); w = uvScale.
        maths::vec4 axisUpWorld{};    // xyz = unit world-space box "up" axis (decal-local +Y);   w = normalStrength.
        maths::vec4 projAxisWorld{};  // xyz = unit world-space projection axis (right x up); w = opacity [0,1].
        maths::vec4 tint{};           // xyz = per-decal tint color multiplier; w = emissive strength (linear HDR).
        maths::vec4 params{};         // x = DecalType (as float); y = sortPriority; z/w reserved (0).
    };
    static_assert(sizeof(DecalInstanceGPU) == 144,
        "DecalInstanceGPU must match DecalInstance in decal_common.glsl exactly (std430 layout)");

    // Everything GenerateShowcaseDecals() hands to renderer::DecalProjectionPass::Init: the GPU
    // instance array plus the parallel world AABBs geometry::BuildDecalBVH needs, index-aligned 1:1
    // (bounds[i] is the AABB of instances[i]).
    struct DecalSceneData {
        std::vector<DecalInstanceGPU> instances;
        std::vector<geometry::DecalBoxBounds> bounds;
    };

    // Human-friendly authoring description of one oriented decal box, converted to a DecalInstanceGPU
    // (+ its world AABB) by MakeDecalInstance below. `right`/`up` must be unit and mutually
    // orthogonal; the projection axis is derived as right x up.
    struct DecalBoxDesc {
        maths::vec3 center{};
        maths::vec3 right{ 1.0f, 0.0f, 0.0f };
        maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        maths::vec3 halfExtents{ 1.0f, 1.0f, 1.0f }; // (along right, along up, along projection axis).
        DecalType type = DecalType::Moss;
        float opacity = 1.0f;      // Master coverage multiplier [0,1].
        float sortPriority = 0.0f; // Higher = applied LATER (wins over lower-priority overlaps), UE decal sort order.
        maths::vec3 tint{ 1.0f, 1.0f, 1.0f };
        float emissive = 0.0f;     // Linear HDR emissive strength (0 = non-emissive).
        float uvScale = 1.0f;      // Multiplies the procedural pattern frequency across the decal face.
        float normalStrength = 1.0f; // Scales the crack/relief normal perturbation written into the GBuffer.
    };

    // Converts one DecalBoxDesc into its GPU instance and its world-space AABB. The local-to-world box
    // transform has columns (right*hx, up*hy, fwd*hz, center); worldToDecal is its inverse. The AABB
    // is the tight span of the box's 8 world-space corners.
    inline void MakeDecalInstance(const DecalBoxDesc& d, DecalInstanceGPU& outInstance,
        geometry::DecalBoxBounds& outBounds) {
        const maths::vec3 right = d.right.Normalize();
        const maths::vec3 up = d.up.Normalize();
        // Projection axis = right x up (points along the box's third dimension -- the "into the
        // surface" direction). DecalProject.comp uses abs(dot(surfaceNormal, projAxis)) for its
        // two-sided angle fade, so this axis's sign is irrelevant to the result (documented there).
        const maths::vec3 fwd = right.Cross(up).Normalize();
        const float hx = d.halfExtents.x, hy = d.halfExtents.y, hz = d.halfExtents.z;

        // decalToWorld, column-major (m[row + col*4], matching maths::mat4's own storage convention).
        maths::mat4 decalToWorld;
        decalToWorld.m[0]  = right.x * hx; decalToWorld.m[1]  = right.y * hx; decalToWorld.m[2]  = right.z * hx; decalToWorld.m[3]  = 0.0f;
        decalToWorld.m[4]  = up.x    * hy; decalToWorld.m[5]  = up.y    * hy; decalToWorld.m[6]  = up.z    * hy; decalToWorld.m[7]  = 0.0f;
        decalToWorld.m[8]  = fwd.x   * hz; decalToWorld.m[9]  = fwd.y   * hz; decalToWorld.m[10] = fwd.z   * hz; decalToWorld.m[11] = 0.0f;
        decalToWorld.m[12] = d.center.x;   decalToWorld.m[13] = d.center.y;   decalToWorld.m[14] = d.center.z;   decalToWorld.m[15] = 1.0f;

        outInstance.worldToDecal = decalToWorld.Inverse();
        outInstance.axisRightWorld = maths::vec4{ right.x, right.y, right.z, d.uvScale };
        outInstance.axisUpWorld = maths::vec4{ up.x, up.y, up.z, d.normalStrength };
        outInstance.projAxisWorld = maths::vec4{ fwd.x, fwd.y, fwd.z, d.opacity };
        outInstance.tint = maths::vec4{ d.tint.x, d.tint.y, d.tint.z, d.emissive };
        outInstance.params = maths::vec4{ static_cast<float>(d.type), d.sortPriority, 0.0f, 0.0f };

        // World AABB over the 8 box corners.
        maths::ResetAABB(outBounds.boundsMin, outBounds.boundsMax);
        for (int sx = -1; sx <= 1; sx += 2) {
            for (int sy = -1; sy <= 1; sy += 2) {
                for (int sz = -1; sz <= 1; sz += 2) {
                    const maths::vec3 corner = d.center
                        + right * (hx * static_cast<float>(sx))
                        + up * (hy * static_cast<float>(sy))
                        + fwd * (hz * static_cast<float>(sz));
                    maths::ExpandAABB(outBounds.boundsMin, outBounds.boundsMax, corner);
                }
            }
        }
    }

    // Hand-authors the fixed showcase decal set placed onto the base gallery scene (see
    // renderer::VulkanContext::GridSlot / GenerateGeometry for that scene's own layout). Decals are
    // deliberately placed on the large matte DIELECTRIC surfaces -- the procedural terrain floor
    // (y ~= -0.8) and the two Lumen/GI Cornell corner walls (red at x = -1.8, green at z = -1.8) --
    // where the albedo de-modulation compositing DecalProject.comp uses is well conditioned (see that
    // shader's own header comment on why metals/emissives are avoided). Exercises all four procedural
    // decal materials plus a deliberate high-priority overlap that demonstrates sort-priority ordering.
    inline DecalSceneData GenerateShowcaseDecals() {
        // Scene anchors, kept in sync with VulkanContext::GenerateGeometry (kFloorTopY / the two wall
        // slot positions). The terrain surface undulates +/- ~0.4 around kFloorTopY, so floor decal
        // boxes are made generously deep (half-depth ~0.8) to fully enclose that variation.
        constexpr float kFloorTopY = -0.8f;
        constexpr float kFloorDecalHalfDepth = 0.9f;
        const maths::vec3 kFloorRight{ 1.0f, 0.0f, 0.0f };
        const maths::vec3 kFloorUp{ 0.0f, 0.0f, 1.0f }; // Maps world +Z onto the decal's local +Y; right x up = (0,-1,0) (down into floor).

        std::vector<DecalBoxDesc> descs;

        // 1) Floor MOSS patch -- near the MegaLights / translucent open floor area.
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ 2.6f, kFloorTopY, 2.6f };
            d.right = kFloorRight; d.up = kFloorUp;
            d.halfExtents = maths::vec3{ 2.4f, 2.4f, kFloorDecalHalfDepth };
            d.type = DecalType::Moss;
            d.opacity = 0.95f; d.sortPriority = 0.0f;
            d.tint = maths::vec3{ 0.55f, 0.85f, 0.45f };
            d.uvScale = 1.0f;
            descs.push_back(d);
        }
        // 2) Floor CRACKS -- dielectric-plane / pyramid zone floor.
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ -2.6f, kFloorTopY, 2.4f };
            d.right = kFloorRight; d.up = kFloorUp;
            d.halfExtents = maths::vec3{ 2.0f, 2.0f, kFloorDecalHalfDepth };
            d.type = DecalType::Cracks;
            d.opacity = 1.0f; d.sortPriority = 1.0f;
            d.tint = maths::vec3{ 0.9f, 0.9f, 0.9f };
            d.uvScale = 1.2f; d.normalStrength = 1.0f;
            descs.push_back(d);
        }
        // 3) Floor PAINT hazard marking -- centered in front of the Lumen/GI corner, faintly emissive
        //    so it reads as a painted guideline.
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ 0.0f, kFloorTopY, 2.4f };
            d.right = kFloorRight; d.up = kFloorUp;
            d.halfExtents = maths::vec3{ 1.7f, 1.7f, kFloorDecalHalfDepth };
            d.type = DecalType::Paint;
            d.opacity = 1.0f; d.sortPriority = 2.0f;
            d.tint = maths::vec3{ 1.0f, 0.75f, 0.05f }; // Warning-yellow.
            d.emissive = 0.6f; d.uvScale = 1.0f;
            descs.push_back(d);
        }
        // 4) High-priority PAINT overlap ON TOP of the moss patch (1) -- deliberately overlaps decal 1
        //    with a higher sortPriority to demonstrate priority ordering (paint wins over moss where
        //    they overlap, exactly UE decal sort behavior).
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ 2.6f, kFloorTopY, 2.6f };
            d.right = kFloorRight; d.up = kFloorUp;
            d.halfExtents = maths::vec3{ 1.1f, 1.1f, kFloorDecalHalfDepth };
            d.type = DecalType::Paint;
            d.opacity = 1.0f; d.sortPriority = 5.0f; // Highest -> applied last -> wins the overlap.
            d.tint = maths::vec3{ 0.95f, 0.15f, 0.15f }; // Red marking.
            d.emissive = 0.0f; d.uvScale = 1.4f;
            descs.push_back(d);
        }
        // 5) Wall A (red, x = -1.8, faces +X) RUST stain. Box "up" = world +Y, "right" = world +Z;
        //    right x up = (-1,0,0), i.e. projecting along -X into the wall.
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ -1.78f, 0.7f, 0.0f };
            d.right = maths::vec3{ 0.0f, 0.0f, 1.0f };
            d.up = maths::vec3{ 0.0f, 1.0f, 0.0f };
            d.halfExtents = maths::vec3{ 1.3f, 1.3f, 0.35f };
            d.type = DecalType::Rust;
            d.opacity = 1.0f; d.sortPriority = 1.0f;
            d.tint = maths::vec3{ 0.7f, 0.35f, 0.15f };
            d.uvScale = 1.1f;
            descs.push_back(d);
        }
        // 6) Wall B (green, z = -1.8, faces +Z) MOSS. Box "up" = world +Y, "right" = world +X;
        //    right x up = (0,0,1) (two-sided angle fade makes the sign irrelevant, see MakeDecalInstance).
        {
            DecalBoxDesc d{};
            d.center = maths::vec3{ 0.0f, 0.7f, -1.78f };
            d.right = maths::vec3{ 1.0f, 0.0f, 0.0f };
            d.up = maths::vec3{ 0.0f, 1.0f, 0.0f };
            d.halfExtents = maths::vec3{ 1.3f, 1.3f, 0.35f };
            d.type = DecalType::Moss;
            d.opacity = 0.9f; d.sortPriority = 1.0f;
            d.tint = maths::vec3{ 0.45f, 0.75f, 0.40f };
            d.uvScale = 1.3f;
            descs.push_back(d);
        }

        DecalSceneData data;
        data.instances.resize(descs.size());
        data.bounds.resize(descs.size());
        for (size_t i = 0; i < descs.size(); ++i) {
            MakeDecalInstance(descs[i], data.instances[i], data.bounds[i]);
        }
        return data;
    }

}
