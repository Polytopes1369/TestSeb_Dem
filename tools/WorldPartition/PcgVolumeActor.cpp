#include "PcgVolumeActor.h"

namespace worldpartition {

    namespace {

        // Property keys a PcgVolume's data is encoded under -- named constants (not inline string
        // literals at each use site) so BuildPcgVolumeActorRecord and TryParsePcgVolumeDesc can
        // never drift onto mismatched key strings.
        constexpr const char* kPropKeyGraphAssetPath = "PcgGraphAssetPath";
        constexpr const char* kPropKeySeed = "PcgSeed";

    } // namespace

    ActorRecord BuildPcgVolumeActorRecord(const Uuid& uuid, const PcgVolumeDesc& desc) {
        ActorRecord record;
        record.uuid = uuid;
        record.parentUuid = kNilUuid; // PCG Volumes are always root-level placements, never attached/parented.
        record.className = kPcgVolumeClassName;
        record.actorLabel = "PcgVolume";

        // Identity transform: `desc.bounds` IS the world-space region (per PcgVolumeDesc's own
        // comment), so localBounds/transform are set such that RecomputeWorldBounds() reproduces it
        // exactly -- no rotation, no scale, no translation to fold in or lose precision to.
        record.transform = ActorTransform{};
        record.localBounds = desc.bounds;
        record.RecomputeWorldBounds();

        record.streamingFlags = ActorStreamingFlags::SpatiallyLoaded;

        record.properties.push_back({ kPropKeyGraphAssetPath, PropertyValue{ desc.graphAssetPath } });
        // PropertyValue has no unsigned-integer alternative (see OfpaActor.h's own PropertyValue
        // comment on why it is a small, closed variant); reinterpreted through int32_t instead --
        // C++20-and-later guarantees unsigned<->signed conversion here is exactly a 2's-complement
        // bit-pattern reinterpretation in both directions (see TryParsePcgVolumeDesc's inverse
        // cast), so this round-trips every possible uint32_t value losslessly, not just the ones
        // that happen to fit in int32_t's positive range.
        record.properties.push_back({ kPropKeySeed, PropertyValue{ static_cast<int32_t>(desc.seed) } });

        return record;
    }

    bool TryParsePcgVolumeDesc(const ActorRecord& record, PcgVolumeDesc& outDesc) {
        if (record.className != kPcgVolumeClassName) return false;

        bool foundGraphAssetPath = false;
        bool foundSeed = false;
        std::string graphAssetPath;
        int32_t seedBits = 0;

        for (const PropertyEntry& entry : record.properties) {
            if (entry.key == kPropKeyGraphAssetPath) {
                if (!std::holds_alternative<std::string>(entry.value)) return false; // Wrong-type property: malformed record, fail closed rather than guess.
                graphAssetPath = std::get<std::string>(entry.value);
                foundGraphAssetPath = true;
            } else if (entry.key == kPropKeySeed) {
                if (!std::holds_alternative<int32_t>(entry.value)) return false;
                seedBits = std::get<int32_t>(entry.value);
                foundSeed = true;
            }
        }

        if (!foundGraphAssetPath || !foundSeed) return false; // Missing an expected property: not a (complete) PcgVolume record.

        // `record.worldBounds` (not localBounds): the canonical, transform-resolved spatial field
        // every other reader of an ActorRecord (SceneIndexEntry, SpatialHashGrid::Build) already
        // keys off of -- see this file's own TryParsePcgVolumeDesc header comment.
        outDesc.bounds = record.worldBounds;
        outDesc.graphAssetPath = std::move(graphAssetPath);
        outDesc.seed = static_cast<uint32_t>(seedBits);
        return true;
    }

    std::vector<CellCoord> ComputeOverlappingCells(const AABB& worldBounds, float cellSize) {
        // Reuses SpatialHashGrid's own WorldToCell floor(worldPos / cellSize) conversion (rather
        // than re-deriving it here) so this function can never silently drift onto a different
        // rounding rule than every other actor's cell bucketing in this offline toolset -- see this
        // file's own header comment. Grid2D matches the runtime streaming grid's own ground-plane
        // convention (Y collapsed, CellCoord::y always 0): a PCG Volume's height above/below the
        // ground plane never changes which cell(s) it generates content for, exactly like every
        // other actor already bucketed by BakeDemoWorld.cpp's own SpatialHashGrid.
        const SpatialHashGrid grid(cellSize, GridDimension::Grid2D);

        const CellCoord cellMin = grid.WorldToCell(worldBounds.boundsMin);
        const CellCoord cellMax = grid.WorldToCell(worldBounds.boundsMax);

        // Guard against an inverted/degenerate bounds where boundsMin's cell ends up past
        // boundsMax's cell on some axis (e.g. a caller-supplied AABB with boundsMin > boundsMax) --
        // treat that the same as SpatialHashGrid::Build implicitly would (an empty range on that
        // axis contributes zero cells) rather than looping backwards or underflowing the reserve()
        // below.
        if (cellMin.x > cellMax.x || cellMin.z > cellMax.z) return {};

        std::vector<CellCoord> result;
        result.reserve(static_cast<size_t>(cellMax.x - cellMin.x + 1) * static_cast<size_t>(cellMax.z - cellMin.z + 1));
        for (int32_t cz = cellMin.z; cz <= cellMax.z; ++cz) {
            for (int32_t cx = cellMin.x; cx <= cellMax.x; ++cx) {
                result.push_back(CellCoord{ cx, 0, cz });
            }
        }
        return result;
    }

}
