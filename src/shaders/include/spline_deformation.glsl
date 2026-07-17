#ifndef SPLINE_DEFORMATION_GLSL
#define SPLINE_DEFORMATION_GLSL

// Phase 1 (Nanite advanced) -- runtime spline-driven deformation, UE5.8 Spline Mesh Component
// style: an already-resident entity's REST-POSE LOCAL geometry is bent along a small, static,
// authored Hermite curve at runtime (not a cook-time procedural spline-mesh generator, which was
// explicitly ruled out in favor of this). Applied in LOCAL space, BEFORE the per-entity rigid
// rotation (renderer::EntityRotationCPU/EntityTransform) -- a spline bend is an intrinsic shape
// property of the mesh, exactly like a real Spline Mesh Component's own local-space deformation,
// so it composes underneath whatever rigid spin the entity separately has.
//
// Gated per-entity via ENTITY_FLAG_HAS_SPLINE_DEFORMATION (struct_custo.glsl) so only the demo
// entity (Tube, meshID 6 -- literally a hollow pipe shape, the natural bend target) pays the cost.
//
// SPLINE_MAX_DEVIATION is this feature's equivalent of wpo_deformation.glsl's maxWPOAmplitude
// contract: the culling/LOD-error bound this feature's flag adds (ClusterDAGScreenError.comp,
// ClusterLODCompact.comp) must never be exceeded by the true worst-case displacement between a
// vertex's bent position and its original rest-pose position, or bent geometry could pop through a
// bounding volume the culling pass already decided was safely outside the frustum/occluded. Unlike
// a plain sine sway, a Hermite curve can overshoot its control points (it is not convex-hull-bounded
// the way a Bezier curve is), so this constant is deliberately generous and cross-checked empirically
// at Debug startup by ClusterRenderPipeline's ValidateSplineBounds() (C++ mirror of the math below),
// not derived from this file alone -- see that function's own comment for the validation approach.
// ValidateSplineBounds()'s dense sample of the actual authored curve (ClusterRenderPipeline.cpp's
// kSplineControlPoints) measured a true worst case of ~1.4917 (at the curve's t~=0.944 near the P3
// end, where the bend combines with the tube's own outer-radius cross-section offset) -- 1.6
// keeps a deliberate ~0.1 safety margin above that measured worst case.
#define SPLINE_MAX_DEVIATION 1.6

// The authored demo curve has 4 control points (3 Hermite segments). Entity 6 (Tube)'s rest-pose
// local geometry spans Y in [-SPLINE_REST_POSE_HALF_HEIGHT, +SPLINE_REST_POSE_HALF_HEIGHT] (Height
// = 1.4, recentered by geom_tube.comp's worldOffsetY = -height/2 at generation time) -- the curve
// parameter below reparameterizes exactly that range.
#define SPLINE_CONTROL_POINT_COUNT 4u
#define SPLINE_REST_POSE_HALF_HEIGHT 0.7

// One Hermite control point: a position and an (un-normalized, magnitude matters) tangent vector.
// 32 bytes, std430 -- each vec3's implicit alignment already pads it to 16 bytes, so the explicit
// pad floats add zero hidden slack, matching this codebase's own "keep sizeof() honest" convention
// (see e.g. ClusterRenderPipeline.cpp's WPOGlobalsUBO comment).
struct SplineControlPoint {
    vec3 position;
    float _pad0;
    vec3 tangent;
    float _pad1;
};

// Cubic Hermite basis, evaluated at a two-endpoint segment (p0/t0 at u=0, p1/t1 at u=1). Returns
// both the interpolated position and the curve's tangent direction at u (the analytic derivative
// w.r.t. u, NOT renormalized to arc length -- only its direction is used by the caller).
void HermiteEvaluate(vec3 p0, vec3 t0, vec3 p1, vec3 t1, float u, out vec3 outPosition, out vec3 outTangent) {
    float u2 = u * u;
    float u3 = u2 * u;

    float h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    float h10 = u3 - 2.0 * u2 + u;
    float h01 = -2.0 * u3 + 3.0 * u2;
    float h11 = u3 - u2;
    outPosition = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1;

    float dh00 = 6.0 * u2 - 6.0 * u;
    float dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    float dh01 = -6.0 * u2 + 6.0 * u;
    float dh11 = 3.0 * u2 - 2.0 * u;
    outTangent = dh00 * p0 + dh10 * t0 + dh01 * p1 + dh11 * t1;
}

// Builds an orthonormal bend frame (right, forward) perpendicular to a curve tangent, with a FIXED
// reference axis (local +X, one of the pipe's own original cross-section axes) rather than a
// separately-authored roll channel -- sufficient for this demo's gently-curving control points,
// where the tangent stays close to the original local +Y axis and never nears parallel to +X. Falls
// back to +Z if the tangent ever DOES near-parallel the primary reference, so the frame never
// degenerates into a zero-length cross product (which would otherwise produce a NaN direction).
void BuildBendFrame(vec3 tangentDir, out vec3 bendRight, out vec3 bendForward) {
    vec3 reference = vec3(1.0, 0.0, 0.0);
    if (abs(dot(tangentDir, reference)) > 0.98) {
        reference = vec3(0.0, 0.0, 1.0);
    }
    bendRight = normalize(cross(tangentDir, reference));
    bendForward = cross(bendRight, tangentDir);
}

// localPos: the vertex's LOCAL-space position (worldPos - EntityTransform.center), i.e. still in
// the entity's own rest-pose frame, before the per-entity rigid rotation is applied by the caller.
// controlPoints: this entity's authored curve, exactly SPLINE_CONTROL_POINT_COUNT entries.
// Returns the bent local-space position: the original cross-section offset (localPos.x, localPos.z)
// carried into the curve's own local bend frame at the matching height, replacing localPos.y's
// straight-axis contribution with the curve's actual (possibly bent) position.
vec3 ApplySplineDeformation(vec3 localPos, SplineControlPoint controlPoints[SPLINE_CONTROL_POINT_COUNT]) {
    float tNormalized = clamp((localPos.y + SPLINE_REST_POSE_HALF_HEIGHT) / (2.0 * SPLINE_REST_POSE_HALF_HEIGHT), 0.0, 1.0);
    float globalT = tNormalized * float(SPLINE_CONTROL_POINT_COUNT - 1u);

    int segmentIndex = int(floor(globalT));
    segmentIndex = clamp(segmentIndex, 0, int(SPLINE_CONTROL_POINT_COUNT) - 2);
    float localU = globalT - float(segmentIndex);

    SplineControlPoint c0 = controlPoints[segmentIndex];
    SplineControlPoint c1 = controlPoints[segmentIndex + 1];

    vec3 curvePos, curveTangent;
    HermiteEvaluate(c0.position, c0.tangent, c1.position, c1.tangent, localU, curvePos, curveTangent);

    vec3 bendRight, bendForward;
    BuildBendFrame(normalize(curveTangent), bendRight, bendForward);

    return curvePos + bendRight * localPos.x + bendForward * localPos.z;
}

#endif // SPLINE_DEFORMATION_GLSL
