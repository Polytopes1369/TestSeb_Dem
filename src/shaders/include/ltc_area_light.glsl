#ifndef LTC_AREA_LIGHT_GLSL
#define LTC_AREA_LIGHT_GLSL

// UE5.8 rendering-parity gap G3 (rect / area light specular): Linearly Transformed Cosines (LTC),
// Heitz, Dupuy, Hill & Neubelt, "Real-Time Polygonal-Light Shading with Linearly Transformed
// Cosines" (SIGGRAPH 2016) -- exactly UE5.8's own analytic rect-light specular technique. A
// point-sample importance estimator (what megalights_ris.glsl uses for point lights) cannot correctly
// integrate an area emitter's specular response: a rough surface sees the WHOLE rect in one glossy
// highlight, which a single stochastic point on the rect converges to only very slowly and noisily.
// LTC instead integrates the polygon ANALYTICALLY against a clamped-cosine distribution that has been
// linearly transformed to match the GGX specular lobe -- one closed-form evaluation, zero noise.
//
// --- Why the transform matrix is computed ANALYTICALLY here, not sampled from a LUT ---
// The canonical LTC implementation samples the fitted M^-1 transform (and a magnitude/Fresnel scale)
// from two small precomputed lookup TEXTURES baked offline by numerical GGX fitting. This project
// FORBIDS every baked data/texture asset in the executable (CLAUDE.md hard rule: 100% procedural, no
// loaded texture files). So the genuine, mathematically-exact part of LTC -- the polygon edge
// integration (LTC_IntegrateEdgeVec / LTC_Evaluate below) -- is kept verbatim, and only the fitted
// transform matrix is replaced by LTC_ApproxMinv: a compact CLOSED-FORM approximation of the offline
// GGX fit as a function of (roughness, view angle). It is deliberately an approximation, not the
// exact tabulated fit, and is documented as such -- it reproduces the correct qualitative behavior
// (an identity transform at full roughness where GGX ~= a clamped cosine, a strongly tangentially-
// compressed transform at low roughness giving a sharp highlight, and a grazing-angle skew that tilts
// the lobe toward the horizon) without a single byte of baked data.
//
// Reuses include/ggx_brdf.glsl's F_Schlick verbatim (no duplicate Fresnel math -- that file is this
// codebase's one and only BRDF-building-block header, see its own comment) and include/math_utils.glsl
// for PI. include/megalights_types.glsl provides the MegaLight struct / MegaLightRectTwoSided helper
// the convenience wrapper at the bottom consumes.

#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"
#include "include/megalights_types.glsl"

// Integrate one polygon edge against the clamped-cosine distribution (Heitz et al. 2016, sec 4.1).
// Returns the edge's contribution to the "vector irradiance": summing this over every edge and
// dotting the result with the shading normal (the +z axis in the LTC-transformed frame) yields 2*PI
// times the polygon's clamped-cosine form factor. The rational polynomial for theta/sin(theta) is the
// paper's own low-cost fit, chosen so no acos()/normalize() is needed on the hot path.
vec3 LTC_IntegrateEdgeVec(vec3 v1, vec3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float vv = a / b;
    float thetaSinTheta = (x > 0.0) ? vv : (0.5 * inversesqrt(max(1.0 - x * x, 1.0e-7)) - vv);
    return cross(v1, v2) * thetaSinTheta;
}

// Analytic LTC transform matrix (M^-1) for isotropic GGX -- see this file's own header comment on why
// this closed form replaces the offline-fitted LUT. `NdotV` is the clamped view cosine, `roughness`
// the perceptual roughness. In the shading frame (N = +z, V in the x-z plane) the fitted GGX M^-1 has
// the well-known sparse structure [[m00,0,m02],[0,m11,0],[m20,0,m22]]; here m11 == m00 (isotropic),
// m22 == 1, m20 == 0, and only the tangential scale + a single grazing skew term are modeled.
mat3 LTC_ApproxMinv(float NdotV, float roughness) {
    float a = clamp(roughness, 0.02, 1.0);
    float theta = acos(clamp(NdotV, 0.0, 1.0));
    // Tangential scale: near 0 for a mirror (an extremely narrow lobe -> only near-mirror polygon
    // directions survive the transform -> a crisp highlight), ramping to 1 at full roughness (M^-1 ->
    // identity -> the clamped cosine itself, which is what a maximally-rough GGX lobe approaches).
    float tangentialScale = mix(0.08, 1.0, a);
    // Grazing skew: tilts the transformed lobe toward the horizon as the view grazes, and vanishes as
    // roughness grows (a broad lobe barely skews). Placed in the (row 0, col 2) cross term.
    float skew = sin(theta) * (1.0 - a) * 0.85;
    // Column-major mat3(col0, col1, col2).
    return mat3(
        vec3(tangentialScale, 0.0,             0.0),
        vec3(0.0,             tangentialScale, 0.0),
        vec3(skew,            0.0,             1.0));
}

// Evaluate the clamped-cosine form factor of the quad `points[0..3]` (world space, wound around the
// emitter's front face) as seen from shading point `P` with normal `N` and view `V`, under the LTC
// transform `Minv`. This is the exact LTC polygon integration (Heitz et al. 2016, sec 5) -- the only
// data-free substitution vs. the reference is the horizon-clip form-factor scale, which the reference
// samples from its second LUT and which is replaced here by the closed-form vector-irradiance term
// max(vsum.z, 0) / (2*PI) (the standard analytic form factor, exact for a polygon fully above the
// horizon and a smooth, artifact-free underestimate as it dips below). A two-sided emitter uses
// abs(vsum.z) so its back face lights surfaces too.
float LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided) {
    // Orthonormal shading basis with V in the x-z plane, then fold it into the LTC transform so the
    // polygon is expressed directly in the transformed clamped-cosine space.
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);
    mat3 M = Minv * transpose(mat3(T1, T2, N));

    vec3 L0 = normalize(M * (points[0] - P));
    vec3 L1 = normalize(M * (points[1] - P));
    vec3 L2 = normalize(M * (points[2] - P));
    vec3 L3 = normalize(M * (points[3] - P));

    vec3 vsum = vec3(0.0);
    vsum += LTC_IntegrateEdgeVec(L0, L1);
    vsum += LTC_IntegrateEdgeVec(L1, L2);
    vsum += LTC_IntegrateEdgeVec(L2, L3);
    vsum += LTC_IntegrateEdgeVec(L3, L0);

    float formFactor = twoSided ? abs(vsum.z) : max(vsum.z, 0.0);
    return formFactor / (2.0 * PI);
}

// Convenience wrapper: the analytic GGX specular response of a MEGALIGHT_TYPE_RECT emitter at shading
// point `worldPos`, BEFORE the light's own color/intensity/distance-window (the caller applies those,
// matching every other MegaLights shading term's "caller multiplies by the light's radiometric
// scale" contract). `F0`/`roughness` come from the shading point's resolved Substrate base slab. The
// rect's 4 corners are built from its center (light.position), normal (light.direction), first
// in-plane axis (light.tangentU) and half extents; the second in-plane axis is derived as
// cross(normal, tangentU), exactly as renderer::geometry::LightBVH's own ComputeLightAABB derives it.
vec3 EvaluateRectLTCSpecular(MegaLight light, vec3 N, vec3 V, vec3 worldPos, vec3 F0, float roughness) {
    vec3 n = normalize(light.direction);
    vec3 u = normalize(light.tangentU);
    vec3 vAxis = cross(n, u);
    vec3 hu = u * light.rectHalfExtentX;
    vec3 hv = vAxis * light.rectHalfExtentY;
    vec3 c = light.position;

    vec3 pts[4];
    pts[0] = c - hu - hv;
    pts[1] = c + hu - hv;
    pts[2] = c + hu + hv;
    pts[3] = c - hu + hv;

    float NdotV = clamp(dot(N, V), 1.0e-4, 1.0);
    mat3 Minv = LTC_ApproxMinv(NdotV, roughness);
    float spec = LTC_Evaluate(N, V, worldPos, Minv, pts, MegaLightRectTwoSided(light));
    // Schlick Fresnel at the view angle (ggx_brdf.glsl) -- the LTC magnitude LUT the reference uses
    // additionally tabulates a roughness/Fresnel scale; here the plain NdotV Schlick term is the
    // data-free stand-in, consistent with EvaluateSubstrateReflectionWeight's own Fresnel-only choice.
    vec3 F = F_Schlick(F0, NdotV);
    return F * spec;
}

#endif // LTC_AREA_LIGHT_GLSL
