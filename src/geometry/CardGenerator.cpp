#include "geometry/CardGenerator.h"
#include "core/Logger.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

namespace geometry {

    namespace {

        // Converts a world-space footprint extent into an atlas texel count, clamped to the
        // sampleable/anti-monopolization range declared in CardGenerator.h.
        uint32_t FootprintToTexels(float worldExtent) {
            float texels = std::ceil(worldExtent * kCardTexelsPerWorldUnit);
            texels = std::clamp(texels,
                static_cast<float>(kMinCardResolution),
                static_cast<float>(kMaxCardResolution));
            return static_cast<uint32_t>(texels);
        }

        // The two AABB extents a given face direction projects onto its card plane, in the fixed
        // (cardU, cardV) order the capture shader also uses:
        //   +/-X face -> (extentZ, extentY)
        //   +/-Y face -> (extentX, extentZ)
        //   +/-Z face -> (extentX, extentY)
        // Keeping this mapping in exactly one place (here and mirrored in the GLSL card
        // projection) is what makes CPU-authored UVs and GPU capture/sampling agree.
        void FaceFootprint(uint32_t faceDirection, const maths::vec3& extent,
            float& outU, float& outV) {
            switch (faceDirection) {
            case kCardFacePosX:
            case kCardFaceNegX:
                outU = extent.z; outV = extent.y; break;
            case kCardFacePosY:
            case kCardFaceNegY:
                outU = extent.x; outV = extent.z; break;
            default: // kCardFacePosZ / kCardFaceNegZ
                outU = extent.x; outV = extent.y; break;
            }
        }

    } // namespace

    std::vector<SurfaceCacheCardEntry> GenerateEntityCards(
        uint32_t entityID, const maths::vec3& boundsMin, const maths::vec3& boundsMax) {

        std::vector<SurfaceCacheCardEntry> cards;
        cards.reserve(kMaxCardsPerEntity);

        const maths::vec3 extent = boundsMax - boundsMin;

        for (uint32_t face = 0; face < kMaxCardsPerEntity; ++face) {
            float footprintU = 0.0f;
            float footprintV = 0.0f;
            FaceFootprint(face, extent, footprintU, footprintV);

            // A face seeing a near-zero-area footprint (e.g. the +/-X faces of a flat ground
            // plane) captures nothing usable -- skip it, which is why an entity has "up to",
            // not "exactly", 6 cards.
            if (footprintU < kDegenerateCardExtent || footprintV < kDegenerateCardExtent) {
                continue;
            }

            SurfaceCacheCardEntry card{};
            card.entityID = entityID;
            card.faceDirection = face;
            card.localBoundsMin[0] = boundsMin.x;
            card.localBoundsMin[1] = boundsMin.y;
            card.localBoundsMin[2] = boundsMin.z;
            card.localBoundsMax[0] = boundsMax.x;
            card.localBoundsMax[1] = boundsMax.y;
            card.localBoundsMax[2] = boundsMax.z;
            card.atlasSize[0] = FootprintToTexels(footprintU);
            card.atlasSize[1] = FootprintToTexels(footprintV);
            // atlasOffset / uvMin / uvMax stay zeroed until PackCardsIntoAtlas stamps them.
            cards.push_back(card);
        }

        return cards;
    }

    bool PackCardsIntoAtlas(std::vector<SurfaceCacheCardEntry>& cards, uint32_t atlasSize) {
        if (cards.empty()) {
            return true;
        }

        LOG_INFO(std::format("[CardGenerator] Packing {} cards into a {}x{} atlas...", cards.size(), atlasSize, atlasSize));

        // Shelf packing over an index permutation (tallest card first, so shelf height waste is
        // minimized) -- the cards vector itself keeps its caller-defined order, because the
        // on-disk table's order is part of the format contract (entity-grouped, face-ordered).
        std::vector<uint32_t> order(cards.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&cards](uint32_t a, uint32_t b) {
            if (cards[a].atlasSize[1] != cards[b].atlasSize[1]) {
                return cards[a].atlasSize[1] > cards[b].atlasSize[1];
            }
            return cards[a].atlasSize[0] > cards[b].atlasSize[0]; // Tie-break for determinism.
            });

        const float invAtlasSize = 1.0f / static_cast<float>(atlasSize);

        uint32_t shelfX = 0;       // Next free X inside the current shelf.
        uint32_t shelfY = 0;       // Top of the current shelf.
        uint32_t shelfHeight = 0;  // Height of the current shelf (tallest card placed in it).

        for (uint32_t cardIndex : order) {
            SurfaceCacheCardEntry& card = cards[cardIndex];
            // Every placement reserves the card plus its gutter on the right/bottom, so two
            // adjacent cards always keep >= kCardGutterTexels empty texels between them.
            const uint32_t paddedW = card.atlasSize[0] + kCardGutterTexels;
            const uint32_t paddedH = card.atlasSize[1] + kCardGutterTexels;

            if (paddedW > atlasSize || paddedH > atlasSize) {
                return false; // Single card larger than the whole atlas: mistuned constants.
            }

            if (shelfX + paddedW > atlasSize) {
                // Current shelf is full: open a new shelf above it.
                shelfY += shelfHeight;
                shelfX = 0;
                shelfHeight = 0;
            }
            if (shelfY + paddedH > atlasSize) {
                LOG_ERROR(std::format("[CardGenerator] Atlas exhausted while packing! Failed on card with size {}x{}.", card.atlasSize[0], card.atlasSize[1]));
                return false; // Atlas exhausted.
            }

            card.atlasOffset[0] = shelfX;
            card.atlasOffset[1] = shelfY;
            card.uvMin[0] = static_cast<float>(card.atlasOffset[0]) * invAtlasSize;
            card.uvMin[1] = static_cast<float>(card.atlasOffset[1]) * invAtlasSize;
            card.uvMax[0] = static_cast<float>(card.atlasOffset[0] + card.atlasSize[0]) * invAtlasSize;
            card.uvMax[1] = static_cast<float>(card.atlasOffset[1] + card.atlasSize[1]) * invAtlasSize;

            shelfX += paddedW;
            // Descending-height order guarantees the first card of a shelf is its tallest, but
            // std::max keeps the invariant explicit rather than implied.
            shelfHeight = std::max(shelfHeight, paddedH);
        }

        LOG_INFO(std::format("[CardGenerator] Packed {} cards successfully. Final height usage: {}/{} texels.", cards.size(), shelfY + shelfHeight, atlasSize));
        return true;
    }

}
