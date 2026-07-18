#pragma once

// PCG framework roadmap, Phase 2.3 ("Volume Sampler"): UE5.8 PCG "Volume Sampler" node parity --
// scatters pcg::PcgPoint instances through a 3D volume region (pcg::PcgVolumeData, an oriented
// bounding region -- an OBB, degenerating to a plain AABB when the volume's orientation is
// identity; see PcgSpatialData.h), for use cases like volumetric debris, floating particles-as-
// props, or 3D density fields -- as opposed to the Surface/Terrain samplers (Phase 2.1/2.2,
// PcgSurfaceSampler.h/PcgTerrainSampler.h) which are inherently 2D-on-a-surface. This phase is
// deliberately scoped to sampling only: no Filter/Spawner/graph-engine logic (later phases) lives
// here, matching PcgSpatialData.h's own "data-model layer only" scoping precedent.
//
// Two sampling modes, both implemented in the volume's LOCAL space (i.e. the box's own
// [-halfExtents, +halfExtents] frame) and only transformed into world space as the LAST step
// (worldPos = center + orientation.RotateVector(localPos)). This is deliberate, not incidental:
// local space is the OBB's natural rectangular parametrization -- a regular lattice or an
// independent-per-axis uniform-random sample is trivial and correct there, and is only correct
// there. Building the lattice/samples directly against the volume's WORLD-SPACE AABB instead
// would silently ignore `orientation` for any rotated volume (the lattice would align to world
// axes, not the box's own axes) -- exactly the bug this file's own unit test
// (tests/PcgVolumeSamplerTests.cpp) has a dedicated rotated-OBB case to catch.
//
//   - Grid: a regular 3D lattice of points at a caller-specified world-space spacing (interpreted
//     along the volume's own local axes), with optional per-axis jitter (a seeded-random fraction
//     of each lattice cell's own size) so the result reads as "scattered" rather than a visibly
//     rigid grid, while still guaranteeing every point stays inside the volume.
//   - Random: pure uniform-random points, independently sampled per local axis in
//     [-halfExtents, +halfExtents], with the point COUNT itself driven by a target density
//     (points per cubic meter) times the volume's total volume (8 * halfExtents.x * halfExtents.y
//     * halfExtents.z -- an OBB's volume, in its own local frame, is invariant under rotation).
//
// Determinism: every random draw this sampler performs -- grid jitter offsets, random-mode
// positions, and each output point's own PcgPoint::seed -- is derived from a single
// pcg::PcgSeededRandom stream (or a direct PcgHashCombine call) constructed from the caller's
// `seed` parameter, per PcgSeededRandom.h's hard project-wide rule ("never a global RNG"). Given
// the same (volume, params, seed) triple, repeated SampleVolume() calls produce a byte-identical
// output vector -- see tests/PcgVolumeSamplerTests.cpp's determinism check.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // Selects which of the two sampling strategies SampleVolume() below uses.
    enum class PcgVolumeSamplingMode : uint8_t {
        Grid,   // Regular 3D lattice, optional per-axis jitter -- see PcgVolumeSamplerParams::gridSpacing/jitterFraction.
        Random, // Pure uniform-random points, count driven by PcgVolumeSamplerParams::density.
    };

    // Parameters controlling SampleVolume()'s output. Only the fields relevant to the selected
    // `mode` are consulted (Grid mode ignores `density`; Random mode ignores `gridSpacing`/
    // `jitterFraction`) -- both groups are kept on one struct rather than split into a
    // mode-specific union/variant, matching this codebase's general preference for plain,
    // inspectable POD-ish parameter structs over more elaborate polymorphism for a
    // two-and-only-two-modes case.
    struct PcgVolumeSamplerParams {
        PcgVolumeSamplingMode mode = PcgVolumeSamplingMode::Grid;

        // --- Grid mode ---
        // Spacing between adjacent lattice points along each of the volume's own LOCAL axes
        // (i.e. this is a distance measured inside the OBB's own rotated frame, not world-space
        // X/Y/Z). Any axis whose spacing is <= 0 is silently clamped to a tiny positive minimum
        // internally (see PcgVolumeSampler.cpp's own kMinGridSpacing) rather than looping forever
        // or dividing by zero.
        maths::vec3 gridSpacing{ 1.0f, 1.0f, 1.0f };

        // Per-axis jitter amount, as a fraction in [0, 1] of one lattice CELL's own size along
        // that axis (i.e. jitterFraction * spacing.axis * 0.5 is the maximum distance -- in
        // either direction -- a jittered point can move off its exact lattice position). 0 means
        // no jitter at all (points sit exactly on the lattice); values above 1 are clamped back
        // to 1 internally. Every jittered (and even every un-jittered) point is additionally
        // clamped to stay strictly inside the volume's own bounds before being emitted -- see
        // this file's own PcgVolumeSampler.cpp comment on kContainmentEpsilon for why that safety
        // clamp exists even for nominally in-range positions.
        float jitterFraction = 0.0f;

        // --- Random mode ---
        // Target point density in points per cubic meter. The number of points SampleVolume()
        // emits in Random mode is density * (8 * halfExtents.x * halfExtents.y * halfExtents.z)
        // (the volume's own total volume), rounded to the nearest integer. A density <= 0
        // (or a degenerate zero-volume box) yields zero points, not an error.
        float density = 1.0f;
    };

    // Scatters points through `volume` according to `params`, using `seed` to derive every
    // random decision this call makes (see this header's own top comment for the determinism
    // guarantee). Every returned point's `position` lies inside `volume` (i.e.
    // volume.ContainsWorldPoint(point.position) is true for every element -- verified by
    // tests/PcgVolumeSamplerTests.cpp for both modes and for a rotated OBB); every point's
    // `rotation` defaults to the volume's own `orientation` (a reasonable default for a
    // volumetric prop scattered inside an oriented region -- a future PCG Filter/Transform-node
    // phase can randomize or override this per point, not this sampler's job); `scale`/`density`/
    // `color`/`boundsMin`/`boundsMax`/`steepness` are left at PcgPoint's own documented defaults.
    std::vector<PcgPoint> SampleVolume(const PcgVolumeData& volume, const PcgVolumeSamplerParams& params, uint32_t seed);

}
