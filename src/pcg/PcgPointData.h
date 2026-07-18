#pragma once

// PCG framework roadmap, Phase 1 ("PCG Data Model"): the core PCG data unit, modeled on UE5.8's
// FPCGPoint -- a single procedurally-placed "thing" (a tree, a rock, a grass tuft, ...) carrying a
// full transform plus the extra per-point attributes later PCG graph nodes (samplers/filters/
// spawners/graph engine, all future phases) read and mutate: density, a color/tint, a deterministic
// seed, a local-space bounds, and a steepness (density-falloff-near-edge) factor.
//
// Two representations are provided, exactly mirroring the existing renderer::GpuParticle /
// ParticleCommon.glsl `Particle` split (src/renderer/passes/ParticleSystemPass.h,
// src/shaders/include/ParticleCommon.glsl):
//   - PcgPoint: the CPU-side, human/tooling-facing struct. Uses this project's real math types
//     (maths::vec3/vec4/quat/mat4, core/maths/Maths.h) directly -- no parallel math library.
//   - GpuPcgPoint: a byte-for-byte, std430-compatible mirror suitable for a future PCG points SSBO.
//     Uses FLAT scalar fields (positionX/Y/Z, not a maths::vec3 member) throughout, matching
//     GpuParticle's own established convention -- maths::vec3 has no alignas(16) and is 12 bytes
//     with no implicit padding, so relying on it as a struct member would NOT reproduce std430's
//     mandatory 16-byte vec3 alignment; flat floats make the byte layout obvious by inspection and
//     let ToGpuPoint() below build it with simple field assignment, no reinterpret_cast games.
// The static_assert at the bottom cross-checks GpuPcgPoint's size against pcg_common.glsl's
// PcgGpuPoint struct (src/shaders/include/pcg_common.glsl) -- the two must be kept byte-for-byte in
// sync, exactly as ParticleSystemPass.h's own GpuParticle/EmitterParams static_asserts already
// require for the particle system.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/maths/Maths.h"

namespace pcg {

    // CPU-side point data -- UE5.8 FPCGPoint parity. Every field below has a documented purpose;
    // none are placeholders.
    struct PcgPoint {
        // --- Transform (position/rotation/scale), reusing this project's existing math types. ---
        maths::vec3 position{ 0.0f, 0.0f, 0.0f };
        maths::quat rotation{}; // Identity by default (maths::quat's own default: w=1, x=y=z=0).
        maths::vec3 scale{ 1.0f, 1.0f, 1.0f };

        // --- Density: [0,1], UE5.8 FPCGPoint::Density parity. This is the point's INTRINSIC
        // density value BEFORE the steepness-driven bounds-edge falloff below is applied -- callers
        // wanting the falloff-adjusted value should call GetEffectiveDensity(), not read this field
        // raw, once they have a local-space sample position to test against `boundsMin`/`boundsMax`.
        float density = 1.0f;

        // --- Color/tint: RGBA, UE5.8 FPCGPoint::Color parity (used by e.g. per-instance tint
        // variation on procedurally scattered foliage). maths::vec4 (added to Maths.h by this
        // phase, see that header's own comment) rather than 4 separate floats, since this is the
        // CPU-facing struct where ergonomics matter more than manual std430 byte-layout control.
        maths::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f };

        // --- Seed: deterministic per-point PRNG seed, UE5.8 FPCGPoint::Seed parity. Every future
        // PCG node that needs randomness FOR THIS SPECIFIC POINT (jitter, per-point variation, ...)
        // must derive its own PcgSeededRandom stream from this seed (see PcgSeededRandom.h) --
        // never from a shared/global RNG -- so re-running the same graph on the same input always
        // reproduces the exact same point set (this codebase's hard "the show must play back
        // identically every run" requirement).
        uint32_t seed = 0;

        // --- Bounds: local-space AABB (min/max corners), UE5.8 FPCGPoint::BoundsMin/BoundsMax
        // parity. Local to this point's OWN transform -- i.e. GetLocalToWorld() maps these corners
        // into their actual world-space extent. Defaults to a unit cube centered on the point.
        maths::vec3 boundsMin{ -0.5f, -0.5f, -0.5f };
        maths::vec3 boundsMax{ 0.5f, 0.5f, 0.5f };

        // --- Steepness: [0,1], UE5.8 FPCGPoint::Steepness parity.
        //
        // Interpretation (this is a faithful best-effort reproduction of UE5.8's documented PCG
        // steepness semantics, not a literal source port -- Epic's own PCG framework source is not
        // available to this codebase, so the exact edge-blend curve is this phase's own
        // implementation of the documented behavior): steepness controls how sharply this point's
        // density falls off as a sample position approaches the local-space bounds edge.
        //   steepness = 0 -> "soft": density decreases smoothly and gradually starting from the
        //     point's own center (a broad, gentle gradient spanning the ENTIRE bounds extent).
        //   steepness = 1 -> "hard": density stays at its full value everywhere inside the bounds,
        //     dropping to 0 only in an infinitesimally thin band right at the outer edge (reads as
        //     a hard-edged box rather than a soft blob).
        // Values in between linearly interpolate the position, within the bounds, at which the
        // falloff ramp BEGINS (see ComputeDensityFalloff's own comment for the exact formula).
        float steepness = 0.0f;

        // Composes this point's local-to-world matrix: Translate(position) * FromQuat(rotation) *
        // Scale(scale) -- UE5.8 FPCGPoint::Transform parity (a single FTransform there; this
        // codebase has no dedicated "transform" struct, so the matrix is composed directly).
        maths::mat4 GetLocalToWorld() const {
            return maths::mat4::Translate(position) * maths::mat4::FromQuat(rotation) * maths::mat4::Scale(scale);
        }

        // Chebyshev (box-distance) edge factor for a LOCAL-space sample position: 0.0 at this
        // point's local-space bounds center, 1.0 exactly at the bounds edge, >1.0 outside the
        // bounds entirely. Uses per-axis distance normalized by that axis' own half-extent (so a
        // non-cubic bounds box is handled correctly, not just a uniform cube) and takes the max
        // across axes (Chebyshev/L-infinity distance) since the bounds is an AABB, not a sphere --
        // matching a box's own natural "nearest face" distance metric.
        float ComputeEdgeFactor(const maths::vec3& localPos) const {
            const maths::vec3 halfExtent = (boundsMax - boundsMin) * 0.5f;
            const maths::vec3 center = (boundsMin + boundsMax) * 0.5f;
            const maths::vec3 offset = localPos - center;

            constexpr float kMinExtent = 1.0e-6f; // Guards a degenerate (zero-width) bounds axis from a divide-by-zero.
            const float ex = std::abs(offset.x) / std::max(halfExtent.x, kMinExtent);
            const float ey = std::abs(offset.y) / std::max(halfExtent.y, kMinExtent);
            const float ez = std::abs(offset.z) / std::max(halfExtent.z, kMinExtent);
            return std::max({ ex, ey, ez });
        }

        // Steepness-driven density multiplier in [0,1] for a LOCAL-space sample position -- see
        // this struct's own `steepness` field comment for the semantics being implemented.
        // `threshold` (== steepness) is the edge-factor value below which density stays at full
        // strength (1.0); beyond it, density ramps linearly down to 0 by the time edgeFactor
        // reaches 1.0 (the bounds edge itself). steepness=1 => threshold=1 => the ramp only has
        // room to act in the infinitesimal [1,1] band at the very edge (a hard box); steepness=0 =>
        // threshold=0 => the ramp spans the point's ENTIRE bounds, from center to edge (a soft,
        // linear gradient).
        float ComputeDensityFalloff(const maths::vec3& localPos) const {
            const float edgeFactor = ComputeEdgeFactor(localPos);
            const float threshold = std::clamp(steepness, 0.0f, 1.0f);
            if (edgeFactor <= threshold) {
                return 1.0f;
            }
            constexpr float kMinDenom = 1.0e-6f; // Guards threshold==1 (denom would otherwise be exactly 0) from a divide-by-zero.
            const float denom = std::max(1.0f - threshold, kMinDenom);
            const float falloff = 1.0f - (edgeFactor - threshold) / denom;
            return std::clamp(falloff, 0.0f, 1.0f);
        }

        // This point's actual density at a LOCAL-space sample position: intrinsic `density` scaled
        // by the steepness-driven edge falloff above. This is the value a future PCG density-filter
        // node (later phase) should read, not the raw `density` field.
        float GetEffectiveDensity(const maths::vec3& localPos) const {
            return std::clamp(density, 0.0f, 1.0f) * ComputeDensityFalloff(localPos);
        }
    };

    // GPU-mirror of PcgPoint -- byte-for-byte std430 layout, matching pcg_common.glsl's
    // PcgGpuPoint struct exactly (see that file's own comment for the full field-packing
    // rationale: every vec3 there is immediately followed by a scalar that packs into the SAME
    // 16-byte std430 slot, the identical "vec3 + trailing scalar" idiom ParticleCommon.glsl's
    // `Particle` struct already established for this codebase).
    struct GpuPcgPoint {
        float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f;
        float density = 1.0f;                                              // Packs into position's 16-byte slot.
        float rotationX = 0.0f, rotationY = 0.0f, rotationZ = 0.0f, rotationW = 1.0f;
        float scaleX = 1.0f, scaleY = 1.0f, scaleZ = 1.0f;
        float steepness = 0.0f;                                            // Packs into scale's 16-byte slot.
        float colorR = 1.0f, colorG = 1.0f, colorB = 1.0f, colorA = 1.0f;
        float boundsMinX = -0.5f, boundsMinY = -0.5f, boundsMinZ = -0.5f;
        uint32_t seed = 0;                                                 // Packs into boundsMin's 16-byte slot.
        float boundsMaxX = 0.5f, boundsMaxY = 0.5f, boundsMaxZ = 0.5f;
        float _pad0 = 0.0f;                                                // Packs into boundsMax's 16-byte slot; reserved for future per-point authoring data (e.g. a future custom-attribute-set index), matching GpuParticle's own reserved-pad convention.
    };
    // 6 x 16-byte std430 slots: position+density, rotation, scale+steepness, color, boundsMin+seed,
    // boundsMax+pad. Cross-checked against pcg_common.glsl's PcgGpuPoint by
    // PcgDataModelSmokeTest.cpp's own runtime check (GLSL has no static_assert equivalent glslc
    // will enforce for us, so the smoke test's size-consistency check is the practical substitute).
    static_assert(sizeof(GpuPcgPoint) == 96, "GpuPcgPoint must match pcg_common.glsl's PcgGpuPoint struct exactly (std430 layout)");

    // Converts a CPU-side PcgPoint into its GPU-mirror representation -- the one place PcgPoint's
    // maths::vec3/vec4/quat members get unpacked into GpuPcgPoint's flat scalar fields, so future
    // upload code (a Phase 2+ PCG points SSBO writer) has a single, already-tested conversion to
    // call rather than re-deriving the field mapping at every call site.
    inline GpuPcgPoint ToGpuPoint(const PcgPoint& point) {
        GpuPcgPoint gpu;
        gpu.positionX = point.position.x; gpu.positionY = point.position.y; gpu.positionZ = point.position.z;
        gpu.density = point.density;
        gpu.rotationX = point.rotation.x; gpu.rotationY = point.rotation.y; gpu.rotationZ = point.rotation.z; gpu.rotationW = point.rotation.w;
        gpu.scaleX = point.scale.x; gpu.scaleY = point.scale.y; gpu.scaleZ = point.scale.z;
        gpu.steepness = point.steepness;
        gpu.colorR = point.color.x; gpu.colorG = point.color.y; gpu.colorB = point.color.z; gpu.colorA = point.color.w;
        gpu.boundsMinX = point.boundsMin.x; gpu.boundsMinY = point.boundsMin.y; gpu.boundsMinZ = point.boundsMin.z;
        gpu.seed = point.seed;
        gpu.boundsMaxX = point.boundsMax.x; gpu.boundsMaxY = point.boundsMax.y; gpu.boundsMaxZ = point.boundsMax.z;
        gpu._pad0 = 0.0f;
        return gpu;
    }

}
