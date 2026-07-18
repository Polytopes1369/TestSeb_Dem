// PCG framework roadmap, Phase 2.3 ("Volume Sampler") -- implementation. See PcgVolumeSampler.h
// for the full design rationale (local-space lattice/sampling, world-space transform as the last
// step, determinism guarantee). This file only fleshes out the two sampling strategies (Grid,
// Random) and the small shared helpers both lean on.

#include "pcg/PcgVolumeSampler.h"

#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cmath>

namespace pcg {

    namespace {

        // Guards a caller-supplied grid spacing of <= 0 from producing an infinite/absurdly huge
        // lattice (division by zero, or by a near-zero value) -- silently floors it to a tiny but
        // strictly positive minimum instead of asserting, since a degenerate spacing on one axis
        // of an otherwise well-formed request is a reasonable thing for a future PCG graph node
        // to produce transiently (e.g. while a user is still dragging a spacing slider).
        constexpr float kMinGridSpacing = 1.0e-4f;

        // Nudges the extent/spacing ratio before flooring it into a lattice point count, so an
        // exactly-divisible ratio (e.g. halfExtent=1.0, spacing=0.5 -> ratio exactly 4.0) is never
        // undercounted due to ordinary floating-point division producing something like
        // 3.999999 instead of 4.0 -- without this nudge that would floor to 3, silently dropping
        // one full row of lattice points for the extremely common "spacing evenly divides the
        // volume" case.
        constexpr float kGridCountFloorEpsilon = 1.0e-4f;

        // Every emitted point's local-space position is clamped to
        // [-(halfExtent - kContainmentEpsilon), +(halfExtent - kContainmentEpsilon)] on each axis
        // as a final safety step before the local-to-world transform, for two independent
        // reasons: (1) grid-mode jitter can otherwise push a boundary-row lattice point's
        // position past the volume's own edge, and (2) PcgVolumeData::ContainsWorldPoint performs
        // a world-space round-trip (RotateVector, then its inverse) that -- for a rotated OBB --
        // can introduce a few ULPs of floating-point error even for a position that started
        // EXACTLY on the local-space boundary; a plain `<=` containment test with zero margin
        // would then occasionally reject a point this sampler itself considers perfectly valid.
        // 1e-4 is generously larger than that rotation round-trip error (empirically ~1e-6 for
        // this codebase's typical world-space magnitudes) while being visually and physically
        // negligible for any real scattering use case.
        constexpr float kContainmentEpsilon = 1.0e-4f;

        float SafeGridSpacing(float spacing) {
            return std::max(spacing, kMinGridSpacing);
        }

        // Number of lattice points along a single axis, spanning [-halfExtent, +halfExtent]
        // inclusive at step `safeSpacing` (already clamped via SafeGridSpacing by the caller) --
        // i.e. floor((2*halfExtent) / safeSpacing) + 1 lattice positions
        // -halfExtent, -halfExtent+safeSpacing, ..., all the way up to (but never past)
        // +halfExtent. Always at least 1 (a single point at the volume's local-space origin
        // offset), even for a degenerate zero-extent axis.
        int32_t ComputeAxisGridPointCount(float halfExtent, float safeSpacing) {
            const float extent = 2.0f * std::max(halfExtent, 0.0f);
            const int32_t count = static_cast<int32_t>(std::floor(extent / safeSpacing + kGridCountFloorEpsilon)) + 1;
            return std::max(count, 1);
        }

        // Clamps a LOCAL-space position into the volume's own bounds, inset by
        // kContainmentEpsilon on every axis (see that constant's own comment above for why the
        // inset exists). Shared by both sampling modes so the containment guarantee is
        // implemented exactly once.
        maths::vec3 ClampToVolumeLocalBounds(const maths::vec3& localPos, const maths::vec3& halfExtents) {
            const auto clampAxis = [](float value, float halfExtent) {
                const float bound = std::max(halfExtent - kContainmentEpsilon, 0.0f);
                return std::clamp(value, -bound, bound);
                };
            return maths::vec3{
                clampAxis(localPos.x, halfExtents.x),
                clampAxis(localPos.y, halfExtents.y),
                clampAxis(localPos.z, halfExtents.z)
            };
        }

        // Transforms a volume-LOCAL-space position into world space: rotate by the volume's own
        // orientation, then offset by its center -- the exact inverse of
        // PcgVolumeData::ContainsWorldPoint's own world-to-local transform (PcgSpatialData.h), so
        // a point built here always round-trips back through ContainsWorldPoint correctly.
        maths::vec3 VolumeLocalToWorld(const PcgVolumeData& volume, const maths::vec3& localPos) {
            return volume.center + volume.orientation.RotateVector(localPos);
        }

        // Builds the one PcgPoint shared by both sampling modes for a given local-space sample
        // position and linear sample index: world position via VolumeLocalToWorld, rotation
        // defaulted to the volume's own orientation (see PcgVolumeSampler.h's own SampleVolume
        // comment), and a deterministic per-point seed via a direct PcgHashCombine(seed,
        // sampleIndex) -- NOT sequentially drawn from a shared PcgSeededRandom stream, so a
        // point's seed depends only on its own index, never on how many OTHER random draws (e.g.
        // grid jitter) happened to precede it earlier in iteration order.
        PcgPoint MakeSampledPoint(const PcgVolumeData& volume, const maths::vec3& localPos, uint32_t seed, uint32_t sampleIndex) {
            PcgPoint point;
            point.position = VolumeLocalToWorld(volume, ClampToVolumeLocalBounds(localPos, volume.halfExtents));
            point.rotation = volume.orientation;
            point.seed = PcgHashCombine(seed, sampleIndex);
            return point;
        }

        // --- Grid mode -----------------------------------------------------------------------
        std::vector<PcgPoint> SampleVolumeGrid(const PcgVolumeData& volume, const PcgVolumeSamplerParams& params, uint32_t seed) {
            const float spacingX = SafeGridSpacing(params.gridSpacing.x);
            const float spacingY = SafeGridSpacing(params.gridSpacing.y);
            const float spacingZ = SafeGridSpacing(params.gridSpacing.z);

            const int32_t countX = ComputeAxisGridPointCount(volume.halfExtents.x, spacingX);
            const int32_t countY = ComputeAxisGridPointCount(volume.halfExtents.y, spacingY);
            const int32_t countZ = ComputeAxisGridPointCount(volume.halfExtents.z, spacingZ);

            const float jitterFraction = std::clamp(params.jitterFraction, 0.0f, 1.0f);

            std::vector<PcgPoint> points;
            points.reserve(static_cast<size_t>(countX) * static_cast<size_t>(countY) * static_cast<size_t>(countZ));

            // One shared stream for every jitter draw this call makes, consumed in a fixed
            // (ix outer, iy middle, iz inner) nested-loop order -- deterministic since the loop
            // bounds themselves are a pure function of (volume, params).
            PcgSeededRandom jitterStream(seed);
            uint32_t sampleIndex = 0;

            for (int32_t ix = 0; ix < countX; ++ix) {
                const float baseX = -volume.halfExtents.x + static_cast<float>(ix) * spacingX;
                for (int32_t iy = 0; iy < countY; ++iy) {
                    const float baseY = -volume.halfExtents.y + static_cast<float>(iy) * spacingY;
                    for (int32_t iz = 0; iz < countZ; ++iz) {
                        const float baseZ = -volume.halfExtents.z + static_cast<float>(iz) * spacingZ;

                        maths::vec3 localPos{ baseX, baseY, baseZ };
                        if (jitterFraction > 0.0f) {
                            // Each jitter draw is at most half a cell's own size in either
                            // direction, scaled by jitterFraction -- so a fully-jittered point
                            // (jitterFraction == 1) can reach, but never cross, the midpoint
                            // between itself and an adjacent lattice position.
                            const float jx = jitterStream.NextFloatRange(-1.0f, 1.0f) * jitterFraction * spacingX * 0.5f;
                            const float jy = jitterStream.NextFloatRange(-1.0f, 1.0f) * jitterFraction * spacingY * 0.5f;
                            const float jz = jitterStream.NextFloatRange(-1.0f, 1.0f) * jitterFraction * spacingZ * 0.5f;
                            localPos = localPos + maths::vec3{ jx, jy, jz };
                        }

                        points.push_back(MakeSampledPoint(volume, localPos, seed, sampleIndex));
                        ++sampleIndex;
                    }
                }
            }
            return points;
        }

        // --- Random mode ----------------------------------------------------------------------
        std::vector<PcgPoint> SampleVolumeRandom(const PcgVolumeData& volume, const PcgVolumeSamplerParams& params, uint32_t seed) {
            const float totalVolume = 8.0f
                * std::max(volume.halfExtents.x, 0.0f)
                * std::max(volume.halfExtents.y, 0.0f)
                * std::max(volume.halfExtents.z, 0.0f);
            const float expectedCountF = std::max(params.density, 0.0f) * totalVolume;
            // Round-to-nearest (via double to avoid any float-precision surprise right at a .5
            // boundary) rather than truncate, so a density/volume combination that lands very
            // close to an integer target isn't systematically undercounted.
            const size_t sampleCount = static_cast<size_t>(std::llround(static_cast<double>(expectedCountF)));

            std::vector<PcgPoint> points;
            points.reserve(sampleCount);

            PcgSeededRandom rng(seed);
            for (size_t i = 0; i < sampleCount; ++i) {
                // Independent per-axis uniform sample in local space -- correct here because
                // local space IS the OBB's natural rectangular parametrization (see this file's
                // header comment); sampling directly against a rotated volume's WORLD-space AABB
                // instead would both ignore orientation and waste samples outside the OBB.
                const float lx = rng.NextFloatRange(-volume.halfExtents.x, volume.halfExtents.x);
                const float ly = rng.NextFloatRange(-volume.halfExtents.y, volume.halfExtents.y);
                const float lz = rng.NextFloatRange(-volume.halfExtents.z, volume.halfExtents.z);

                points.push_back(MakeSampledPoint(volume, maths::vec3{ lx, ly, lz }, seed, static_cast<uint32_t>(i)));
            }
            return points;
        }

    } // namespace

    std::vector<PcgPoint> SampleVolume(const PcgVolumeData& volume, const PcgVolumeSamplerParams& params, uint32_t seed) {
        switch (params.mode) {
        case PcgVolumeSamplingMode::Grid:
            return SampleVolumeGrid(volume, params, seed);
        case PcgVolumeSamplingMode::Random:
            return SampleVolumeRandom(volume, params, seed);
        }
        // Unreachable for a valid PcgVolumeSamplingMode value; keeps MSVC's /W4
        // "not all control paths return a value" warning quiet without a fallthrough default
        // case that would otherwise silently swallow a future third mode's enum value.
        return {};
    }

}
