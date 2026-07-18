#pragma once
// Phase A of the MegaLights native-port roadmap (see the approved plan,
// C:\Users\Seb\.claude\plans\memoized-splashing-seal.md): a growable, bindless-style point-light
// population for stochastic RIS-weighted direct lighting (renderer::MegaLightsPass), independent
// from renderer::LightingTypes.h's SceneLights/kMaxPointLights (that representation stays exactly
// as-is -- it feeds only renderer::SurfaceCachePass's own capture pipeline, a genuinely different
// consumer with a deliberately small UBO-sized cap; see this file's own GenerateProceduralLights()
// comment for why the two representations are not unified).

#include <array>
#include <cstdint>

#include "core/maths/Maths.h"

namespace renderer {

    // "UE5.8 showcase scale" -- the user's explicit Phase A choice (see the approved plan). Growable
    // in a later phase simply by raising this constant; RIS iterates g_Lights.lightCount at
    // runtime, no shader-side format change is needed.
    constexpr uint32_t kMaxMegaLights = 256u;

    // Niagara-parity render-integration roadmap, D4 (particles as light emitters): a small, FIXED
    // number of extra light SLOTS reserved at the tail of MegaLightsPass' own light SSBO -- see
    // MegaLightsPass::GetParticleLightsBufferOffset()'s own comment. Deliberately NOT folded into
    // kMaxMegaLights itself (which stays the STATIC procedural population's own exact size,
    // GenerateProceduralLights() below is completely unaware this capacity even exists) -- these
    // slots are instead re-written every frame by renderer::ParticleSystemPass::RecordExtractLights
    // from a bounded sample of currently-alive emissive particles, giving nearby OPAQUE geometry
    // (lit via MegaLightsPass' existing RIS + shadow-visibility-ray pipeline, completely unmodified
    // otherwise) a real, GPU-driven way to be lit by bright particles (e.g. embers) without any
    // CPU readback and without touching the static population's own generation/BVH-build code path
    // beyond appending this many placeholder (always-candidate, zero-weight-until-written) leaf
    // entries once at Init() time -- see MegaLightsPass.cpp's own STEP 1/1.5 comment for the full
    // derivation of why a placeholder-AABB approach is required for the BVH-biased opaque shading
    // path specifically, not just the SSBO content itself.
    constexpr uint32_t kMaxParticleDerivedLights = 16u;

    // UE5.8 rendering-parity gap G3 (extended MegaLights light-type roster): the light-shape
    // discriminator carried in MegaLight::lightType below. A POINT light ignores every extended field
    // (direction/tangentU/cone/rect params) and behaves exactly as the pre-G3 isotropic point light
    // did -- so every existing procedurally-scattered light and every particle-derived tail slot
    // (ParticleLightExtract.comp) that only ever authored {position,radius,color,intensity} stays
    // bit-identical in behavior with lightType left at its default 0. Mirrored value-for-value by
    // megalights_types.glsl's own MEGALIGHT_TYPE_* constants.
    enum class MegaLightType : uint32_t {
        Point = 0u,        // Isotropic inverse-square point light (pre-G3 behavior, the default).
        Spot = 1u,         // Cone-windowed: candela inverse-square + smoothstep(inner,outer) cone falloff.
        Rect = 2u,         // Area (rect) emitter: LTC analytic specular + stochastic point-on-rect diffuse.
        Photometric = 3u,  // Point light shaped by a procedural parametric "IES-style" angular profile.
    };

    // Procedural photometric ("IES-style") angular-falloff shapes -- a small set of PLAIN ANALYTIC
    // profiles selectable per MegaLightType::Photometric light, approximating what a baked .ies file
    // buys (non-uniform angular shaping) with ZERO baked data (CLAUDE.md hard rule: no data assets in
    // the .exe). Stored in the low byte of MegaLight::iesProfileAndFlags; mirrored by
    // megalights_types.glsl's own MEGALIGHT_IES_* constants.
    enum class MegaLightIESProfile : uint32_t {
        Narrow = 0u,   // Tight high-intensity beam (a narrow-cone spot-like pencil).
        Wide = 1u,     // Broad, near-uniform flood over the forward hemisphere.
        BarnDoor = 2u, // Asymmetric rectangular "barn-door" cut (wide on one in-plane axis, narrow on the other).
    };

    // Bit 8 of MegaLight::iesProfileAndFlags: a MegaLightType::Rect light emits from BOTH faces when
    // set, only from the +direction face (one-sided, the physical default for a real softbox) when
    // clear.
    constexpr uint32_t kMegaLightFlagRectTwoSided = 1u << 8;

    // GLSL-friendly, std430-compatible mirror of MegaLight in
    // src/shaders/include/megalights_types.glsl -- 80 bytes, five {vec3, scalar} blocks, same
    // field-ordering convention as renderer::MaterialParameters (every vec3 immediately followed by
    // the scalar filling its std430 base-alignment gap, so no explicit padding is needed).
    //
    // G3 layout note: the first two blocks (position/radius/color/intensity) are byte-for-byte the
    // pre-G3 32-byte struct, so the light SSBO's first 32 bytes of every slot are wire-compatible with
    // any writer that only knows the old layout (the array STRIDE still changes to 80, which is why
    // every SSBO consumer of MegaLight -- all of which #include this file's GLSL mirror transitively
    // -- must be recompiled, but no consumer's field OFFSETS shift). The three extended blocks carry:
    //   direction   -- spot cone axis / rect surface normal / photometric forward axis (points FROM
    //                  the light OUT into the scene, matching DirectionalLight::direction's convention).
    //   tangentU    -- rect first in-plane axis (unit, orthogonal to direction); the rect's second
    //                  in-plane axis is derived on the GPU as cross(direction, tangentU), saving a
    //                  block. Also the barn-door photometric profile's own in-plane reference axis.
    //   spotCosInner/spotCosOuter -- cosines of the inner/outer cone HALF-angles (inner > outer, so
    //                  spotCosInner > spotCosOuter); the shaded cone window is a smoothstep between them.
    //   rectHalfExtentX/rectHalfExtentY -- rect half-width along tangentU / half-height along the
    //                  derived second axis.
    //   iesProfileAndFlags -- MegaLightIESProfile in the low byte, feature flags (kMegaLightFlag*) in
    //                  the high bits.
    struct MegaLight {
        maths::vec3 position{};
        float radius = 1.0f;   // Distance at which the smooth windowed attenuation reaches zero.
        maths::vec3 color = { 1.0f, 1.0f, 1.0f };
        float intensity = 1.0f;
        // --- G3 extended shape parameters (ignored by MegaLightType::Point) ---
        maths::vec3 direction = { 0.0f, -1.0f, 0.0f };
        uint32_t lightType = static_cast<uint32_t>(MegaLightType::Point);
        maths::vec3 tangentU = { 1.0f, 0.0f, 0.0f };
        float rectHalfExtentX = 0.5f;
        float spotCosInner = 0.0f;
        float spotCosOuter = 0.0f;
        float rectHalfExtentY = 0.5f;
        uint32_t iesProfileAndFlags = 0u;
    };
    static_assert(sizeof(MegaLight) == 80,
        "MegaLight must match MegaLight in megalights_types.glsl exactly (std430 layout)");

    struct MegaLightsData {
        std::array<MegaLight, kMaxMegaLights> lights{};
        uint32_t count = 0;
    };

    // Procedurally scatters `kMaxMegaLights` point lights, the large majority concentrated on the
    // dedicated "MegaLights" showcase zone (VulkanContext::GridSlot slot 8, the Cylinder) so
    // MegaLights reads as its own distinct feature rather than a uniform ambient sprinkle across
    // every zone -- the other 11 primitive zones (VulkanContext::GridSlot/kEntityCount --
    // duplicated here as a small local lookup table rather than including VulkanContext.h,
    // matching this codebase's existing renderer/VulkanContext one-directional-boundary
    // convention, e.g. core::EntityTransformCPU's own duplicated rotation math) get only a sparse
    // accent count each. The floor and the 2 static Lumen-corner walls intentionally get none --
    // they're lit by the scene's own sun instead. Deterministic, fixed-seed (never
    // std::random_device) -- matches GenerateShowcaseMaterialTable's own "a demo must look
    // identical every playback" rationale.
    MegaLightsData GenerateProceduralLights(uint32_t seed);

}
