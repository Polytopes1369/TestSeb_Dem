#include "renderer/MegaLightsTypes.h"

#include <cmath>
#include <random>

namespace renderer {

    namespace {

        // Duplicated from VulkanContext::GridSlot() -- see this file's header comment on why (the
        // one-directional renderer/VulkanContext boundary). Must be kept in sync if the demo's grid
        // layout ever changes; kPlacementEntityCount mirrors VulkanContext::kEntityCount.
        constexpr uint32_t kPlacementEntityCount = 13u;
        constexpr float kGridSpacing = 3.0f;

        maths::vec3 EntityGridPosition(uint32_t entityIndex) {
            int col = static_cast<int>(entityIndex) % 3;
            int row = static_cast<int>(entityIndex) / 3;
            return maths::vec3{ (col - 1) * kGridSpacing, 0.0f, (row - 1) * kGridSpacing };
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

        uint32_t baseCountPerEntity = kMaxMegaLights / kPlacementEntityCount;
        uint32_t remainder = kMaxMegaLights % kPlacementEntityCount; // Distributed one-extra-each across the first `remainder` entities.

        uint32_t writeIndex = 0;
        for (uint32_t entityIndex = 0; entityIndex < kPlacementEntityCount; ++entityIndex) {
            uint32_t countThisEntity = baseCountPerEntity + (entityIndex < remainder ? 1u : 0u);
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
