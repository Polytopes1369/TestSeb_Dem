#include "renderer/MegaLightsTypes.h"

#include <array>
#include <cmath>
#include <random>

namespace renderer {

    namespace {

        // Duplicated from VulkanContext::GridSlot() -- see this file's header comment on why (the
        // one-directional renderer/VulkanContext boundary). Must be kept in sync if the gallery
        // layout ever changes; kPlacementEntityCount mirrors the 12 primitive zones only (indices
        // 0..11 of VulkanContext::GridSlot's domain) -- the floor and the 2 static Lumen walls
        // (VulkanContext::kEntityCount == 15) never receive MegaLights lights, see this file's own
        // header comment.
        constexpr uint32_t kPlacementEntityCount = 12u;
        constexpr float kZonePitch = 4.0f;
        constexpr float kPairOffset = 1.0f;

        struct ZoneEntry { float col, row, pairOffset; };
        constexpr std::array<ZoneEntry, kPlacementEntityCount> kLayout = {{
            /* 0  Box (metal A, chrome)      */ {  1.0f, -1.0f, -kPairOffset },
            /* 1  Cone (WPO/displacement)    */ {  0.0f, -1.0f,  0.0f },
            /* 2  Icosphere (Nanite A)       */ { -1.0f, -1.0f, -kPairOffset },
            /* 3  Plane (dielectric A)       */ { -1.0f,  0.0f, -kPairOffset },
            /* 4  Sphere (glass/transparent) */ {  1.0f,  0.0f,  0.0f },
            /* 5  Torus (translucent)        */ { -1.0f,  1.0f,  0.0f },
            /* 6  Tube (emissive)            */ {  0.0f,  1.0f,  0.0f },
            /* 7  Capsule (metal B, gold)    */ {  1.0f, -1.0f,  kPairOffset },
            /* 8  Cylinder (MegaLights hero) */ {  1.0f,  1.0f,  0.0f },
            /* 9  Pyramid (dielectric B)     */ { -1.0f,  0.0f,  kPairOffset },
            /* 10 TorusKnot (Nanite B)       */ { -1.0f, -1.0f,  kPairOffset },
            /* 11 ChamferBox (Lumen/GI hero) */ {  0.0f,  0.0f,  0.0f },
        }};

        // MegaLights showcase zone -- see this file's own header comment for why this one zone
        // gets the large majority of the light budget.
        constexpr uint32_t kMegaLightsZoneIndex = 8u;

        maths::vec3 EntityGridPosition(uint32_t entityIndex) {
            const ZoneEntry& z = kLayout[entityIndex];
            return maths::vec3{ z.col * kZonePitch + z.pairOffset, 0.0f, z.row * kZonePitch };
        }

        // Same HSV -> RGB formula as MaterialParameterTable.h's own local lambda (duplicated rather
        // than shared -- a 15-line color-space conversion is not worth a new shared header for two
        // call sites, matching this codebase's own precedent of small self-contained generators).
        maths::vec3 HsvToRgb(float h, float s, float v) {
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
        }

    } // namespace

    MegaLightsData GenerateProceduralLights(uint32_t seed) {
        MegaLightsData data{};

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);

        // MegaLights showcase zone (the Cylinder, slot 8) gets the large majority of the light
        // budget so the feature reads as its own distinct, densely-lit area; the remaining 11
        // primitive zones share a sparse accent count each -- enough for a specular highlight on
        // the metal/glass zones without competing visually with the MegaLights zone itself.
        // UE5.8 rendering-parity gap G3: reserve the last kTypedLightCount slots of the population for
        // the new SPOT/RECT/PHOTOMETRIC lights appended after this point scatter -- the isotropic
        // point scatter below is capped kTypedLightCount slots short so it never overruns them.
        constexpr uint32_t kSpotCount = 4u;
        constexpr uint32_t kRectCount = 4u;
        constexpr uint32_t kPhotometricCount = 4u;
        constexpr uint32_t kTypedLightCount = kSpotCount + kRectCount + kPhotometricCount;
        constexpr uint32_t kPointLightCapacity = kMaxMegaLights - kTypedLightCount;

        constexpr uint32_t kMegaLightsZoneCount = 200u;
        constexpr uint32_t kAccentEntityCount = kPlacementEntityCount - 1u; // Every zone except MegaLights.
        uint32_t accentBudget = kPointLightCapacity - kMegaLightsZoneCount;
        uint32_t baseCountPerEntity = accentBudget / kAccentEntityCount;
        uint32_t remainder = accentBudget % kAccentEntityCount; // Distributed one-extra-each across the first `remainder` accent entities.

        uint32_t writeIndex = 0;
        uint32_t accentEntitiesSeen = 0;
        for (uint32_t entityIndex = 0; entityIndex < kPlacementEntityCount; ++entityIndex) {
            uint32_t countThisEntity;
            if (entityIndex == kMegaLightsZoneIndex) {
                countThisEntity = kMegaLightsZoneCount;
            } else {
                countThisEntity = baseCountPerEntity + (accentEntitiesSeen < remainder ? 1u : 0u);
                ++accentEntitiesSeen;
            }
            maths::vec3 base = EntityGridPosition(entityIndex);

            for (uint32_t i = 0; i < countThisEntity && writeIndex < kPointLightCapacity; ++i) {
                // Jittered disk offset (sqrt for uniform area density, not uniform radius) around
                // the entity's own grid position -- small enough (~1.2 unit max radius) that lights
                // visibly interact with that entity's own geometry rather than blending into an
                // ambient wash, which is what makes RIS-weighted stochastic sampling actually matter
                // (only a handful of the 256 lights realistically reach any given surface point).
                float angle = unit(rng) * 6.28318530717958647692f;
                float diskRadius = std::sqrt(unit(rng)) * 1.2f;
                float jitterX = std::cos(angle) * diskRadius;
                float jitterZ = std::sin(angle) * diskRadius;
                float jitterY = 0.3f + unit(rng) * 1.7f; // [0.3, 2.0]

                MegaLight light{};
                light.position = base + maths::vec3(jitterX, jitterY, jitterZ);
                light.color = HsvToRgb(unit(rng), 0.75f, 1.0f);
                // Radius scaled well below UE5.8's human-scale-level norm to suit this ~12x12 unit
                // compact demo grid -- see the approved plan's own documented UE5.8-scale adaptation.
                light.radius = 0.8f + unit(rng) * 1.7f;      // [0.8, 2.5]
                // Real luminous intensity in CANDELA (renderer::MegaLight's own comment, 2026-07-17
                // recalibration matching UE5.8's own Point Light unit) -- [200, 800] candela is the
                // same order of magnitude as UE5.8's own default new Point Light (~398 candela, see
                // renderer::PointLight's own comment), scaled per-light since up to 200 of these
                // stack within the dedicated MegaLights showcase zone.
                light.intensity = 200.0f + unit(rng) * 600.0f;

                data.lights[writeIndex] = light;
                ++writeIndex;
            }
        }

        // =====================================================================================
        // UE5.8 rendering-parity gap G3: append a handful of SPOT, RECT (area) and PHOTOMETRIC
        // ("IES-style") lights at named showcase zones. Deterministic placement (never random) so the
        // new types are easy to locate and visually verify rather than lost in the 200-strong
        // hero-zone point cloud. All still within kMaxMegaLights (the point scatter above was capped
        // kTypedLightCount slots short), so the BVH build + SSBO upload in MegaLightsPass::Init cover
        // them with zero further bookkeeping.
        // =====================================================================================
        auto makeTangent = [](const maths::vec3& dir) -> maths::vec3 {
            // Any unit vector orthogonal to `dir` -- pick the world axis least aligned with it to
            // avoid a degenerate cross product (same principle as ggx_brdf.glsl's BuildTangentBasis).
            maths::vec3 ref = (std::abs(dir.y) < 0.99f) ? maths::vec3(0.0f, 1.0f, 0.0f)
                                                        : maths::vec3(1.0f, 0.0f, 0.0f);
            return dir.Cross(ref).Normalize();
        };
        auto cosDeg = [](float deg) { return std::cos(deg * 3.14159265358979323846f / 180.0f); };

        // --- Spot lights: downward cones over four hero zones (candela inverse-square + a smoothstep
        // cone falloff between the inner/outer half-angles, matching UE5.8's own spot cone). ---
        struct SpotSpec { uint32_t zone; float height; float innerDeg; float outerDeg; float hue; float intensity; float range; };
        constexpr std::array<SpotSpec, kSpotCount> kSpots = {{
            {  0u, 3.5f, 14.0f, 26.0f, 0.05f, 1400.0f, 6.0f }, // Box (chrome metal)
            {  4u, 3.5f, 16.0f, 30.0f, 0.55f, 1200.0f, 6.0f }, // Sphere (glass)
            {  7u, 3.2f, 12.0f, 22.0f, 0.12f, 1500.0f, 6.0f }, // Capsule (gold metal)
            { 11u, 4.0f, 18.0f, 34.0f, 0.75f, 1300.0f, 7.0f }, // ChamferBox (Lumen/GI hero)
        }};
        for (const SpotSpec& s : kSpots) {
            if (writeIndex >= kMaxMegaLights) break;
            MegaLight light{};
            light.position = EntityGridPosition(s.zone) + maths::vec3(0.0f, s.height, 0.0f);
            light.direction = maths::vec3(0.0f, -1.0f, 0.0f); // straight down onto the zone
            light.tangentU = maths::vec3(1.0f, 0.0f, 0.0f);
            light.color = HsvToRgb(s.hue, 0.6f, 1.0f);
            light.intensity = s.intensity;
            light.radius = s.range;
            light.spotCosInner = cosDeg(s.innerDeg);
            light.spotCosOuter = cosDeg(s.outerDeg);
            light.lightType = static_cast<uint32_t>(MegaLightType::Spot);
            data.lights[writeIndex++] = light;
        }

        // --- Rect (area) lights: softbox-style panels around the hero zones (LTC analytic specular +
        // stochastic point-on-rect diffuse, evaluated in MegaLightsFinalShade.comp). ---
        struct RectSpec { uint32_t zone; maths::vec3 offset; maths::vec3 normal; float halfW; float halfH; float hue; float intensity; float range; bool twoSided; };
        const std::array<RectSpec, kRectCount> kRects = {{
            { 8u, {  2.2f, 1.6f,  0.0f }, { -1.0f, 0.0f,  0.0f }, 0.9f, 1.2f, 0.58f, 900.0f, 7.0f, false }, // beside MegaLights cylinder, facing -X
            { 8u, { -2.2f, 1.6f,  0.0f }, {  1.0f, 0.0f,  0.0f }, 0.9f, 1.2f, 0.02f, 900.0f, 7.0f, false }, // opposite side, facing +X
            { 4u, {  0.0f, 1.4f,  2.2f }, {  0.0f, 0.0f, -1.0f }, 0.8f, 0.8f, 0.60f, 800.0f, 6.5f, false }, // in front of the glass sphere, facing -Z
            { 5u, {  0.0f, 2.6f,  0.0f }, {  0.0f, -1.0f, 0.0f }, 1.0f, 1.0f, 0.85f, 700.0f, 6.0f, true  }, // overhead panel on the torus, two-sided
        }};
        for (const RectSpec& r : kRects) {
            if (writeIndex >= kMaxMegaLights) break;
            MegaLight light{};
            light.position = EntityGridPosition(r.zone) + r.offset;
            maths::vec3 n = r.normal.Normalize();
            light.direction = n;
            light.tangentU = makeTangent(n);
            light.rectHalfExtentX = r.halfW;
            light.rectHalfExtentY = r.halfH;
            light.color = HsvToRgb(r.hue, 0.5f, 1.0f);
            light.intensity = r.intensity;
            light.radius = r.range;
            light.lightType = static_cast<uint32_t>(MegaLightType::Rect);
            light.iesProfileAndFlags = r.twoSided ? kMegaLightFlagRectTwoSided : 0u;
            data.lights[writeIndex++] = light;
        }

        // --- Photometric ("IES-style") lights: one of each parametric analytic profile (narrow beam,
        // wide flood, asymmetric barn-door) at distinct zones. No baked .ies data (CLAUDE.md rule). ---
        struct PhotoSpec { uint32_t zone; float height; maths::vec3 dir; MegaLightIESProfile profile; float hue; float intensity; float range; };
        const std::array<PhotoSpec, kPhotometricCount> kPhotos = {{
            {  2u, 3.4f, {  0.0f, -1.0f, 0.0f }, MegaLightIESProfile::Narrow,   0.15f, 1600.0f, 6.5f }, // Icosphere (Nanite A): narrow pencil
            { 10u, 3.0f, {  0.0f, -1.0f, 0.0f }, MegaLightIESProfile::Wide,     0.45f,  900.0f, 6.5f }, // TorusKnot (Nanite B): wide flood
            {  6u, 2.8f, {  0.3f, -1.0f, 0.0f }, MegaLightIESProfile::BarnDoor, 0.08f, 1200.0f, 6.0f }, // Tube (emissive): barn-door
            {  9u, 3.2f, {  0.0f, -1.0f, 0.0f }, MegaLightIESProfile::Narrow,   0.68f, 1400.0f, 6.0f }, // Pyramid (dielectric B): narrow pencil
        }};
        for (const PhotoSpec& p : kPhotos) {
            if (writeIndex >= kMaxMegaLights) break;
            MegaLight light{};
            light.position = EntityGridPosition(p.zone) + maths::vec3(0.0f, p.height, 0.0f);
            maths::vec3 d = p.dir.Normalize();
            light.direction = d;
            light.tangentU = makeTangent(d);
            light.color = HsvToRgb(p.hue, 0.55f, 1.0f);
            light.intensity = p.intensity;
            light.radius = p.range;
            light.lightType = static_cast<uint32_t>(MegaLightType::Photometric);
            light.iesProfileAndFlags = static_cast<uint32_t>(p.profile);
            data.lights[writeIndex++] = light;
        }

        data.count = writeIndex;
        return data;
    }

}
