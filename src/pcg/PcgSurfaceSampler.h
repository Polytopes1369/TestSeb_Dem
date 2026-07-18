#pragma once

// PCG framework roadmap, Phase 2.1 ("Surface Sampler"): UE5.8 PCG's "Surface Sampler" node --
// scatters pcg::PcgPoint instances across a mesh surface's triangles at a target density
// (points per square meter), with per-point tangent-plane position jitter, each point projected
// exactly onto its source triangle (barycentric position) with a normal-aligned rotation.
//
// SCOPE BOUNDARY (read this before wiring this sampler into anything): this engine's actual
// triangle data for a resident mesh lives in the Nanite cluster/geometry system (src/geometry/,
// renderer::GpuGeometryPagePool / GeometryStreamingCoordinator) -- GPU-resident cluster pages, not
// something this phase reads back wholesale on the CPU for every sample (that "read back
// GPU-resident Nanite geometry to the CPU" problem is a separate, much bigger concern for a future
// phase/caller, not this one). This phase's actual scope is the sampling ALGORITHM: it accepts an
// explicit CPU-side triangle list (PcgSurfaceTriangle, below) as its real input, built/tested
// against data the CALLER provides. A pcg::PcgSurfaceData reference (meshID/materialID/worldOffset,
// see PcgSpatialData.h) is accepted as an input parameter purely for future wiring -- a later phase
// that DOES resolve PcgSurfaceData against the resident cluster system to produce a
// std::vector<PcgSurfaceTriangle> can hand that list straight to SampleSurfacePoints() below -- but
// this phase's sampling logic itself never touches src/geometry/ or the GPU.
//
// Algorithm (see PcgSurfaceSampler.cpp for the full implementation):
//   1. Compute each triangle's area (0.5 * |cross(B-A, C-A)|).
//   2. Build a running cumulative-area prefix sum across the triangle list (in input order) --
//      area-weighted triangle selection is a std::upper_bound binary search into that prefix sum
//      for a uniform-random value in [0, totalArea), i.e. "running cumulative-area binary search",
//      NOT the alias method (simpler to implement correctly and fast enough: O(log N) per sample,
//      and N is a CPU-side triangle list this phase deliberately keeps modest-sized).
//   3. Expected point count = totalArea * density; the fractional remainder is resolved with one
//      extra seeded coin-flip so the point count's EXPECTATION exactly matches density (see
//      SampleSurfacePoints's own comment for the precise draw).
//   4. Per point: pick a triangle (area-weighted), pick a uniform-random barycentric coordinate on
//      it (standard "reflect if u+v>1" parallelogram-to-triangle folding), interpolate position and
//      normal, align local +Y to the interpolated normal (this codebase's world-up convention is
//      +Y, see core/Camera.cpp's worldUp), derive a fresh per-point seed from the SAME seeded
//      stream (never the raw input seed reused), and optionally jitter the position along the
//      surface's local tangent plane.
//
// Determinism: SampleSurfacePoints() constructs its OWN pcg::PcgSeededRandom stream internally from
// PcgSurfaceSamplerParams::seed -- every random draw the algorithm makes descends from that single
// seed in a fixed, documented order (see the .cpp), so calling it twice with the same triangle list
// and the same params produces byte-identical output. See tests/PcgSurfaceSamplerTests.cpp for the
// proof (determinism, area-weighting statistics, and degenerate/empty/zero-density edge cases).

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // One CPU-side triangle to sample from: 3 world-space vertex positions plus 3 per-vertex
    // normals (interpolated via the sampled barycentric coordinate, "smooth shading" style, exactly
    // like a rasterizer's own vertex-normal interpolation -- NOT re-derived from the face normal on
    // every sample, so callers that already have real vertex normals -- e.g. a Nanite cluster's own
    // per-vertex normal, once a future phase wires this up -- get faithful results). Callers with
    // only a flat face normal (no authored per-vertex normals) can simply set normalA == normalB ==
    // normalC to that face normal; interpolation of three identical vectors reproduces it exactly.
    struct PcgSurfaceTriangle {
        maths::vec3 positionA{ 0.0f, 0.0f, 0.0f };
        maths::vec3 positionB{ 0.0f, 0.0f, 0.0f };
        maths::vec3 positionC{ 0.0f, 0.0f, 0.0f };

        maths::vec3 normalA{ 0.0f, 1.0f, 0.0f };
        maths::vec3 normalB{ 0.0f, 1.0f, 0.0f };
        maths::vec3 normalC{ 0.0f, 1.0f, 0.0f };
    };

    // Sampling parameters -- UE5.8 Surface Sampler node parity (PointsPerSquaredMeter, Seed,
    // PointExtents' jitter counterpart).
    struct PcgSurfaceSamplerParams {
        // Target density in points per square meter of triangle surface area. Non-positive ->
        // SampleSurfacePoints() returns an empty vector (see that function's own edge-case comment).
        float density = 1.0f;

        // Root seed for this sampling call's ENTIRE PcgSeededRandom stream -- every draw (triangle
        // selection, barycentric coordinate, jitter offset, per-point output seed) descends from
        // this single value, in the fixed order documented in PcgSurfaceSampler.cpp. Two calls with
        // the same triangle list + params (this seed included) must produce byte-identical output.
        uint32_t seed = 0;

        // Optional per-point positional jitter, in world-space meters, applied along the sampled
        // point's own LOCAL TANGENT PLANE (i.e. the plane perpendicular to its interpolated
        // surface normal) -- keeps a jittered point near its source triangle rather than lifting it
        // off the surface along the normal. 0.0f (default) disables jitter entirely (the point sits
        // exactly at its sampled barycentric position). This is a simple tangent-plane offset, NOT
        // a re-projection back onto the mesh -- consistent with this phase's "explicit triangle
        // list, no BVH/mesh-query" scope boundary (see this header's own top comment): a jittered
        // point near a triangle edge can therefore end up slightly off the actual mesh surface if
        // the neighboring triangle's plane differs; acceptable for this phase's scope, a future
        // phase's real mesh-query sampler can re-project if that turns out to matter visually.
        float positionJitter = 0.0f;

        // Accepted for future wiring only (see this header's own top-of-file scope-boundary
        // comment) -- NOT read by SampleSurfacePoints()'s sampling algorithm itself in this phase.
        PcgSurfaceData surfaceReference{};
    };

    // Triangle area helper (0.5 * |cross(B-A, C-A)|) -- exposed both because SampleSurfacePoints()
    // needs it internally and because it is independently useful/testable (e.g. the area-weighting
    // statistical test in tests/PcgSurfaceSamplerTests.cpp calls it directly to compute expected
    // ratios). Returns 0.0f (never NaN) for a degenerate (collinear or coincident-vertex) triangle.
    float ComputeTriangleArea(const PcgSurfaceTriangle& triangle);

    // Shortest-arc rotation quaternion that maps local +Y (this codebase's world-up convention, see
    // core/Camera.cpp's worldUp) onto `normal` (assumed unit-length; a zero-length input safely
    // returns the identity quaternion rather than producing a NaN axis). Exposed standalone since it
    // is pure geometry with no RNG/triangle dependency, independently useful/testable.
    maths::quat ComputeUpAlignedRotation(const maths::vec3& normal);

    // Core Surface Sampler algorithm -- see this header's own top-of-file comment for the full
    // algorithm description. Returns an empty vector (never throws, never produces a NaN point) for
    // every degenerate input: an empty triangle list, a triangle list whose triangles are ALL
    // zero-area, or params.density <= 0.0f.
    std::vector<PcgPoint> SampleSurfacePoints(const std::vector<PcgSurfaceTriangle>& triangles, const PcgSurfaceSamplerParams& params);

}
