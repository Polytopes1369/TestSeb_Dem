#ifndef RIVER_SPLINE_GLSL
#define RIVER_SPLINE_GLSL

// Rivers/waterfalls feature (design brief: "un peu de mer, quelques plages, montagnes, plaines,
// rivieres, ruisseau, chutes d'eau, chemins"): single shared source of truth for the authored
// river path -- a 5-control-point polyline in world XZ (this scene's terrain/water both sit at
// worldOffset (0,0,0), so these coordinates ARE world-space) running from a hillside spring, down
// a short, steep cliff (the waterfall), through a lower run, and terminating inside the existing
// flat lake plane (water_params.glsl's kWaterLevel) -- see this file's own kRiverControlHeight
// comment for the exact per-point elevation authoring.
//
// Consumed by THREE independent call sites that must all agree on the exact same path:
//   1. terrain_noise.glsl's SampleTerrainHeight -- carves a channel/valley into the terrain
//      heightfield along this path (RiverChannelMask / RiverBedHeight below).
//   2. geom_river.comp -- generates the actual flowing-water ribbon mesh, chained onto the SAME
//      meshID/entity as the flat lake plane (VulkanContext::GenerateGeometry's WATER block) --
//      see that shader's own header comment for why this deliberately does NOT get a new
//      renderer::EntityData slot.
//   3. WaterForward.frag -- derives per-fragment flow direction (tangent) and slope (for
//      flow-speed/whitewater) directly from world position, entirely analytically (no extra
//      per-vertex attribute exists on geometry::FallbackVertex to carry a baked flow vector
//      through the Fallback Mesh proxy WaterForwardPass actually draws from -- see that struct's
//      own comment in ClusterFormat.h).
//
// Deliberately a PIECEWISE-LINEAR polyline (not the smoother Catmull-Rom curve a purely visual
// spline would use) for every function below: a straight-segment closest-point query is cheap,
// exact, and -- critically -- produces a distance FIELD with no interior discontinuity worse than
// a single crease per interior control point, which stays safe against terrain_noise.glsl's own
// documented DAG-build hazard (see kTerrainRidgeWeight's own comment) because it is smoothed by a
// wide (multi-terrain-vertex-spacing) smoothstep blend radius (kRiverBandOuterHalfWidth), not
// baked as raw high-frequency noise the way the old ridged-noise regression was.
const uint kRiverControlPointCount = 5u;

// World-space (X, Z). P0 = hillside spring (well outside the ~[-5,5] showcase zone grid, see
// VulkanContext::GridSlot's own kZonePitch/kLayout comment, and well inside the 300x300 terrain
// footprint) -> P1 upper course -> P2 cliff edge -> P3 waterfall base (P2->P3 is a short XZ hop
// with a large height drop, see kRiverControlHeight below -- this is what makes that one segment
// read as a near-vertical sheet rather than a gentle rapids) -> P4 mouth, inside the existing lake
// plane (kWaterPlaneSpan/2 == 12, see VulkanContext::GenerateGeometry's WATER block).
const vec2 kRiverControlXZ[5] = vec2[5](
    vec2(30.0, 30.0),
    vec2(18.0, 18.0),
    vec2(12.5, 12.5),
    vec2(12.0, 12.0),
    vec2(6.0,  6.0)
);

// World-space Y (water-surface elevation authored directly, NOT derived from terrain_noise.glsl's
// own ambient fbm -- a river needs a monotonic downhill profile a noise field can't guarantee).
// P0 (2.2) sits well above kFloorTopY (-0.8) +/- kTerrainAmplitude (0.4) -- terrain_noise.glsl's
// RiverChannelMask blends the ambient heightfield UP toward this value near the spline, forming a
// real hill for the spring to sit on, not merely a floating channel. P2->P3 drops 1.8 world units
// over an XZ arc length of ~0.7 (see kRiverControlXZ above) -- a ~68-degree slope, this feature's
// waterfall. P4 (-1.05) sits fractionally BELOW water_params.glsl's kWaterLevel (-1.0) so the
// river's own mouth vertex submerges cleanly under the lake surface instead of leaving a visible
// seam/step at the join.
const float kRiverControlHeight[5] = float[5](2.2, 1.0, 0.8, -1.0, -1.05);

// Channel cross-section: half-width of the flat/near-flat bed (RiverChannelMask == 1 for any point
// within this distance of the centerline), and the additional distance over which the mask fades
// back to 0 (ambient terrain) -- see terrain_noise.glsl's own SampleTerrainHeight for how this
// blend keeps the carve C1-smooth across the terrain's own coarse (4-world-unit) vertex spacing.
const float kRiverBandInnerHalfWidth = 5.0;
const float kRiverBandOuterHalfWidth = 12.0;
// How far BELOW the authored water-surface height (kRiverControlHeight, piecewise-linearly
// interpolated) the carved terrain bed sits at the channel centerline -- guarantees the river
// ribbon mesh (which sits AT the surface height, see geom_river.comp) never pokes through its own
// bed even at the shallow ends of the blend.
const float kRiverBedDepthBelowSurface = 0.6;

// Visual half-width of the water ribbon mesh itself (geom_river.comp) -- narrower than
// kRiverBandInnerHalfWidth so the flowing water reads as a distinct channel inset within its own
// wider carved valley (banks/shore visible on both sides), mirroring a real river's own
// bed-wider-than-water-line profile.
const float kRiverHalfWidth = 2.2;

// Total arc length in CONTROL-POINT-INDEX units (kRiverControlPointCount - 1 segments) -- both
// geom_river.comp and WaterForward.frag parametrize the path by this same continuous `t` so a
// given world position always maps back to a consistent, shared arc coordinate.
const float kRiverArcParamMax = float(kRiverControlPointCount) - 1.0;

// Closest point on the whole polyline to `xz`. Returns the segment-local interpolation weight via
// `outT` (continuous across the whole path, e.g. 2.35 == 35% of the way from P2 to P3),
// `outClosestXZ`/`outHeight` (piecewise-linearly interpolated surface height at that closest
// point), `outTangentXZ` (normalized, direction of increasing t -- "downstream"), and
// `outDistance` (world-space distance from `xz` to the closest point, i.e. this is a genuine
// distance FIELD to the 1D path, used by both the terrain-carve blend mask and the water-shader
// river/lake classification).
void ClosestPointOnRiverPolyline(vec2 xz, out float outT, out vec2 outClosestXZ,
    out float outHeight, out vec2 outTangentXZ, out float outDistance) {
    float bestDistSq = 1.0e30;
    float bestT = 0.0;
    vec2 bestPoint = kRiverControlXZ[0];
    vec2 bestTangent = vec2(1.0, 0.0);
    float bestHeight = kRiverControlHeight[0];

    for (uint i = 0u; i < kRiverControlPointCount - 1u; ++i) {
        vec2 a = kRiverControlXZ[i];
        vec2 b = kRiverControlXZ[i + 1u];
        vec2 ab = b - a;
        float abLenSq = max(dot(ab, ab), 1.0e-8);
        float localT = clamp(dot(xz - a, ab) / abLenSq, 0.0, 1.0);
        vec2 candidate = a + ab * localT;
        float distSq = dot(xz - candidate, xz - candidate);
        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            bestT = float(i) + localT;
            bestPoint = candidate;
            bestTangent = normalize(ab);
            bestHeight = mix(kRiverControlHeight[i], kRiverControlHeight[i + 1u], localT);
        }
    }

    outT = bestT;
    outClosestXZ = bestPoint;
    outHeight = bestHeight;
    outTangentXZ = bestTangent;
    outDistance = sqrt(bestDistSq);
}

// [0, 1] blend weight toward the carved/authored river profile -- 1.0 at/inside the flat bed,
// smoothly 0.0 past the outer band radius. Used by terrain_noise.glsl to blend the ambient
// heightfield toward RiverBedHeight, and by WaterForward.frag to blend flow-mapped shading in
// over the plain flat-lake wave shading.
float RiverChannelMask(vec2 xz) {
    float t, height, distance_;
    vec2 closestXZ, tangent;
    ClosestPointOnRiverPolyline(xz, t, closestXZ, height, tangent, distance_);
    return 1.0 - smoothstep(kRiverBandInnerHalfWidth, kRiverBandOuterHalfWidth, distance_);
}

// Carved terrain bed height at `xz` -- only meaningful where RiverChannelMask(xz) > 0, see this
// function's own caller in terrain_noise.glsl for the actual blend against the ambient fbm.
float RiverBedHeight(vec2 xz) {
    float t, height, distance_;
    vec2 closestXZ, tangent;
    ClosestPointOnRiverPolyline(xz, t, closestXZ, height, tangent, distance_);
    return height - kRiverBedDepthBelowSurface;
}

// Direct evaluation by continuous arc parameter `t` (as opposed to ClosestPointOnRiverPolyline's
// inverse query, given a world XZ position) -- geom_river.comp's own ribbon-mesh generator walks
// `t` directly (one grid axis of its dispatch), so it needs this forward direction, not the
// closest-point search above.
void RiverPointAtT(float t, out vec2 outXZ, out float outHeight, out vec2 outTangentXZ) {
    float tc = clamp(t, 0.0, kRiverArcParamMax);
    uint i = uint(clamp(floor(tc), 0.0, float(kRiverControlPointCount) - 2.0));
    float localT = clamp(tc - float(i), 0.0, 1.0);
    outXZ = mix(kRiverControlXZ[i], kRiverControlXZ[i + 1u], localT);
    outHeight = mix(kRiverControlHeight[i], kRiverControlHeight[i + 1u], localT);
    outTangentXZ = normalize(kRiverControlXZ[i + 1u] - kRiverControlXZ[i]);
}

// World-space position on the water-surface ribbon at arc parameter `t` (downstream) and lateral
// parameter `s` (in [-1, 1], left/right bank) with the given half-width -- the ribbon's centerline
// (s == 0) sits exactly at the authored surface height/position RiverPointAtT returns; the lateral
// axis is the tangent rotated 90 degrees in the XZ plane. geom_river.comp evaluates this at its
// own grid samples AND at small t/s offsets (central difference) to derive the surface normal --
// see that shader's own comment.
vec3 RiverSurfacePosition(float t, float s, float halfWidth) {
    vec2 xz, tangentXZ;
    float height;
    RiverPointAtT(t, xz, height, tangentXZ);
    vec2 lateralXZ = vec2(-tangentXZ.y, tangentXZ.x);
    vec2 offsetXZ = xz + lateralXZ * (s * halfWidth);
    return vec3(offsetXZ.x, height, offsetXZ.y);
}

// Local downhill steepness at arc parameter `t` (world-height-drop per world-XZ-arc-unit,
// unsigned) -- sampled via a small central difference in t. Used to scale both flow-scroll speed
// and the whitewater/foam brightening term (steeper == faster/more turbulent-looking) -- see
// WaterForward.frag's own call site.
float RiverSlopeAtT(float t) {
    float eps = 0.02;
    float t0 = clamp(t - eps, 0.0, kRiverArcParamMax);
    float t1 = clamp(t + eps, 0.0, kRiverArcParamMax);

    uint i0 = uint(clamp(floor(t0), 0.0, float(kRiverControlPointCount) - 2.0));
    float local0 = clamp(t0 - float(i0), 0.0, 1.0);
    float h0 = mix(kRiverControlHeight[i0], kRiverControlHeight[i0 + 1u], local0);
    vec2 p0 = mix(kRiverControlXZ[i0], kRiverControlXZ[i0 + 1u], local0);

    uint i1 = uint(clamp(floor(t1), 0.0, float(kRiverControlPointCount) - 2.0));
    float local1 = clamp(t1 - float(i1), 0.0, 1.0);
    float h1 = mix(kRiverControlHeight[i1], kRiverControlHeight[i1 + 1u], local1);
    vec2 p1 = mix(kRiverControlXZ[i1], kRiverControlXZ[i1 + 1u], local1);

    float arcDist = max(length(p1 - p0), 1.0e-4);
    return abs(h1 - h0) / arcDist;
}

#endif // RIVER_SPLINE_GLSL
