#pragma once

// PCG framework roadmap, Phase 3.1 ("Density Filter, Transform Filter, Attribute Noise/Remap"):
// three small, independent point-filter utilities operating on a std::vector<pcg::PcgPoint> --
// UE5.8 PCG parity for its "Density Filter", "Transform Filter" (position/rotation/scale jitter),
// and "Attribute Noise"/"Density Noise" graph nodes. UE5.8 models each as a separate graph node,
// but they are small enough to share one translation unit here, the same "bundle a params struct +
// a handful of free functions" convention PcgSurfaceSampler.h already uses for its own phase.
//
// SCOPE BOUNDARY: this file is Phase 3.1 only. It does NOT implement self-pruning (Phase 3.2),
// boolean set operations (Phase 3.3), or slope/height filtering (Phase 3.4) -- those are separate,
// independently-developed sibling files (PcgSelfPruningFilter.*/PcgBooleanSetOps.*/
// PcgSlopeHeightFilter.*) and are intentionally out of scope here.
//
// Determinism (this codebase's hard "same input -> byte-identical output, every run" requirement,
// see PcgSeededRandom.h's own header comment): ApplyTransformJitter derives its randomness from
// EACH POINT'S OWN `seed` field (combined with this call's `seed` salt), never from a single shared
// stream walked across the whole batch -- so a given point's jitter is reproducible regardless of
// vector order, batch size, or what other filters already ran on the same points before this one.
// SampleAttributeNoise/ApplyNoiseToDensity are pure functions of (worldPos, frequency, seed) with
// zero internal/static state, so they are trivially deterministic and order-independent by
// construction.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"

namespace pcg {

    // ---------------------------------------------------------------------------------------------
    // 1. Density Filter -- UE5.8 "Density Filter" node parity.
    // ---------------------------------------------------------------------------------------------

    // Keeps only the points whose `density` field falls within [minDensity, maxDensity] (inclusive
    // on BOTH ends). Filters against each point's raw, intrinsic `density` field, NOT
    // GetEffectiveDensity() -- a density-filter graph node has no per-sample local-space position to
    // evaluate the steepness/bounds-edge falloff against (see PcgPointData.h's own PcgPoint::density
    // field comment), exactly matching UE5.8's own Density Filter node, which filters on
    // FPCGPoint::Density directly. Order-preserving: kept points retain their original relative
    // order from `points`. Returns an empty vector (never throws) if no point's density lands in
    // range, including the degenerate case of an empty `points` input.
    std::vector<PcgPoint> FilterByDensity(const std::vector<PcgPoint>& points, float minDensity, float maxDensity);

    // Linearly remaps every point's density from its INPUT range -- [oldMin, oldMax], the actual
    // minimum/maximum `density` value found across `points` -- into [newMin, newMax]. Returns a NEW
    // vector (does not mutate `points`), the same "return a transformed copy" convention
    // FilterByDensity above uses -- Transform Filter (below) is this file's one deliberate exception
    // to that convention, see ApplyTransformJitter's own comment for why in-place mutation makes
    // more sense there specifically.
    //
    // Edge cases: an empty `points` input returns an empty vector. If every point's density is
    // identical (oldMax - oldMin is degenerately ~0, i.e. there is no meaningful "position within
    // the range" to remap), every output point's density is set to exactly `newMin` rather than
    // dividing by ~zero. `newMin`/`newMax` are NOT required to be in [0,1] and are not
    // internally clamped to that range -- this function performs a literal linear remap of whatever
    // range the caller asks for; PcgPoint's own downstream consumers (e.g. GetEffectiveDensity)
    // already clamp density to [0,1] themselves before using it (see PcgPointData.h).
    std::vector<PcgPoint> RemapDensity(const std::vector<PcgPoint>& points, float newMin, float newMax);

    // ---------------------------------------------------------------------------------------------
    // 2. Transform Filter -- UE5.8 "Transform Filter" node parity (jittered position/rotation/scale).
    // ---------------------------------------------------------------------------------------------

    // Independently toggleable position/rotation/scale jitter ranges. Every field left at its
    // default (all three `jitter*` bools false) leaves ApplyTransformJitter() a complete no-op.
    struct PcgTransformJitterParams {
        // Position jitter: a per-axis symmetric range in WORLD-SPACE meters -- each axis draws an
        // independent offset in [-|positionRange.axis|, +|positionRange.axis|] (the absolute value
        // is taken defensively, so a caller passing a negative component still gets a valid
        // symmetric range rather than an inverted/empty one). All-zero components combined with
        // jitterPosition=true is a valid (if pointless) no-op, not a special case.
        bool jitterPosition = false;
        maths::vec3 positionRange{ 0.0f, 0.0f, 0.0f };

        // Rotation jitter: an additional random rotation of up to +/- rotationRangeRadians (a
        // symmetric range, like positionRange) about `rotationAxis` (a WORLD-space axis; does not
        // need to be pre-normalized -- ApplyTransformJitter normalizes it defensively) composed
        // AFTER the point's existing rotation (see ApplyTransformJitter's own comment for the exact
        // composition order and rationale). A zero-length `rotationAxis` combined with
        // jitterRotation=true is treated the same as jitterRotation=false (no meaningful axis to
        // rotate about, so no rotation is applied and no random draw is consumed for it).
        bool jitterRotation = false;
        float rotationRangeRadians = 0.0f;
        maths::vec3 rotationAxis{ 0.0f, 1.0f, 0.0f };

        // Scale jitter: EITHER a single uniform multiplier drawn once from uniformScaleRange and
        // applied identically to all 3 axes (uniformScale=true, the common "vary size, keep
        // proportions" case), OR three independent per-axis multipliers drawn from
        // perAxisScaleRangeMin/Max (uniformScale=false, for deliberately squashed/stretched
        // variation). Both range forms are inclusive MULTIPLIERS applied to the point's EXISTING
        // scale (result = oldScale * drawnMultiplier), not an additive offset -- {1,1} on a range
        // is a no-op for that axis.
        bool jitterScale = false;
        bool uniformScale = true;
        maths::vec2 uniformScaleRange{ 1.0f, 1.0f };
        maths::vec3 perAxisScaleRangeMin{ 1.0f, 1.0f, 1.0f };
        maths::vec3 perAxisScaleRangeMax{ 1.0f, 1.0f, 1.0f };
    };

    // Jitters every point's position/rotation/scale IN PLACE (mutates `points` directly -- unlike
    // FilterByDensity/RemapDensity's "return a new vector" convention, a transform jitter perturbs
    // existing fields on a fixed-size point set with no element added/removed, so mutating in place
    // avoids an unnecessary full-vector copy; callers wanting to preserve the pre-jitter input
    // should copy `points` themselves first, matching std::sort's own in-place convention).
    //
    // `seed` is an ADDITIONAL call-level salt (e.g. lets the same graph re-jitter with a different
    // global variation without touching every point's own PcgPoint::seed field) mixed into each
    // point's own `point.seed` via PcgHashCombine -- it does NOT replace per-point seeding: two
    // points with different `point.seed` values still jitter differently even when `seed` here is
    // identical for both, and re-running this function on the SAME points with the SAME `seed`
    // argument always reproduces the identical jitter output (determinism holds on both axes:
    // per-point AND per-call).
    //
    // Per-point draw order (fixed and documented -- determinism depends on never reordering these):
    //   1. position jitter (3 draws: dx, dy, dz), only if params.jitterPosition.
    //   2. rotation jitter (1 draw: angle), only if params.jitterRotation AND rotationAxis is
    //      non-zero-length.
    //   3. scale jitter (1 draw if params.uniformScale, else 3 draws: mx, my, mz), only if
    //      params.jitterScale.
    // A toggled-off jitter axis consumes ZERO draws from that point's stream (not a draw whose
    // result is simply discarded) -- so e.g. enabling ONLY jitterScale for a point produces the
    // exact same scale multiplier as if jitterPosition/jitterRotation did not exist as concepts at
    // all, independent of what other axes are toggled elsewhere in `params`.
    void ApplyTransformJitter(std::vector<PcgPoint>& points, const PcgTransformJitterParams& params, uint32_t seed);

    // ---------------------------------------------------------------------------------------------
    // 3. Attribute Noise / Remap -- UE5.8 "Attribute Noise" node parity.
    // ---------------------------------------------------------------------------------------------

    // Coherent 3D VALUE noise (not gradient/Perlin noise): samples `worldPos * frequency` into a
    // unit lattice cell, hashes the cell's 8 surrounding integer lattice corners via
    // PcgHashCombine/PcgHash32 (see PcgSeededRandom.h) into independent [0,1) corner values, and
    // trilinearly interpolates between them using a smoothstep (Hermite, 3t^2 - 2t^3) easing curve
    // per axis -- the classic "value noise" construction (Ken Perlin's ORIGINAL 1985 noise was
    // exactly this family, before his 2002 "improved"/gradient-noise variant). Chosen over gradient
    // noise here specifically because it needs no precomputed permutation/gradient table and reuses
    // this codebase's EXISTING PcgHash32/PcgHashCombine hash primitives directly, keeping this file
    // dependency-free beyond PcgSeededRandom.h's hash (see PcgDensityTransformFilter.cpp for the
    // exact lattice-corner-hash + trilinear-blend implementation).
    //
    // Output is always in [0, 1]. Deterministic and stateless: identical (worldPos, frequency, seed)
    // arguments always produce the identical output value on any call, in any order, on any thread
    // -- this is a pure function with zero internal/static state, so it needs no seeded-stream
    // object and no "construct once, call repeatedly" ceremony the way PcgSeededRandom's Next*()
    // methods do.
    float SampleAttributeNoise(const maths::vec3& worldPos, float frequency, uint32_t seed);

    // Concrete usage example wiring SampleAttributeNoise into density: MULTIPLIES (does not
    // overwrite) every point's existing `density` field by a noise sample taken at that point's own
    // WORLD-SPACE `position`, so the visual result reads as "sparser in some regions, denser in
    // others" layered on top of whatever density the point already had (rather than clobbering
    // whatever upstream filter/sampler already wrote there). Result is clamped to [0, 1]
    // (PcgPoint::density's own documented valid range, see PcgPointData.h). Mutates `points` in
    // place, the same in-place convention ApplyTransformJitter above uses (again: a fixed-size,
    // per-point-field perturbation, not an element-count-changing operation).
    void ApplyNoiseToDensity(std::vector<PcgPoint>& points, float frequency, uint32_t seed);

}
