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
        constexpr uint32_t kMegaLightsZoneCount = 200u;
        constexpr uint32_t kAccentEntityCount = kPlacementEntityCount - 1u; // Every zone except MegaLights.
        uint32_t accentBudget = kMaxMegaLights - kMegaLightsZoneCount;
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

            for (uint32_t i = 0; i < countThisEntity && writeIndex < kMaxMegaLights; ++i) {
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
                light.intensity = 0.6f + unit(rng) * 1.9f;   // [0.6, 2.5], modest since 256 stack.

                data.lights[writeIndex] = light;
                ++writeIndex;
            }
        }

        data.count = writeIndex;
        return data;
    }

}
