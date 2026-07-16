#ifndef GGX_BRDF_GLSL
#define GGX_BRDF_GLSL

// Phase 2 (UE5.8 parity roadmap): standard microfacet specular BRDF math, written from scratch --
// a 2026-07-16 audit confirmed zero GGX/BRDF/Fresnel/NDF math existed anywhere in this codebase
// before this file. Consumed by ReflectionTrace.comp (VNDF importance sampling, BuildTangentBasis +
// SampleGGXVNDF) and ReflectionGather.comp (Fresnel-only composite weight, F_Schlick -- a
// deliberately simplified single-sample "Lumen Performance mode" weighting per this phase's approved
// plan, not a full unbiased D*G2/G1 Monte Carlo estimator). D_GGX/G_SmithGGXCorrelated are provided
// here as complete, correctly-named reference BRDF terms (CLAUDE.md rule 1 -- no partial/approximate
// formulas) for a future refinement phase to consume; nothing in the current Phase 2 pipeline calls
// them yet. Every formula below is the well-known textbook/paper version, named explicitly so
// nothing here is an approximation of convenience:
//   - D_GGX: Trowbridge-Reitz normal distribution function.
//   - G_SmithGGXCorrelated: Smith height-correlated visibility term (Heitz, "Understanding the
//     Masking-Shadowing Function in Microfacet-Based BRDFs", 2014) -- already divided by the
//     4*NdotV*NdotL denominator, i.e. this returns VISIBILITY (V), not the raw geometry term (G).
//   - F_Schlick: Schlick's Fresnel approximation.
//   - SampleGGXVNDF: sampling the GGX distribution of visible normals (Heitz, "Sampling the GGX
//     Distribution of Visible Normals", 2018) -- lower variance at grazing angles than sampling
//     the raw NDF, standard for real-time single-sample specular importance sampling.

// PI/TAU come from the existing shared include/math_utils.glsl (const float, not a macro) --
// deliberately NOT redefined here: a `#define PI ...` in this file would textually mangle that
// file's `const float PI = ...` declaration into a syntax error for any shader that includes both.
#include "include/math_utils.glsl"

// Trowbridge-Reitz / GGX normal distribution function.
float D_GGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1.0e-7);
}

// Smith height-correlated visibility term V = G / (4*NdotV*NdotL), Heitz 2014.
float G_SmithGGXCorrelated(float NdotV, float NdotL, float roughness) {
    float a2 = (roughness * roughness) * (roughness * roughness);
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1.0e-5);
}

// Schlick Fresnel. F0 is the surface's reflectance at normal incidence -- callers derive it from
// the metallic workflow: F0 = mix(vec3(0.04), albedo, metallic) (0.04 is the standard dielectric
// baseline, ~4% reflectance for common non-metals like plastic/stone).
vec3 F_Schlick(vec3 F0, float VdotH) {
    float f = pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
    return F0 + (vec3(1.0) - F0) * f;
}

// Branchless orthonormal basis from a unit normal (Duff et al., "Building an Orthonormal Basis,
// Revisited", 2017) -- same battle-tested construction this codebase's own
// SurfaceCacheGIInject.comp already uses inline for its cosine-weighted sampler (not shared from
// there -- see this file's own header comment on why a fresh, self-contained copy lives here
// instead of refactoring that already-working file).
void BuildTangentBasis(vec3 normal, out vec3 tangent, out vec3 bitangent) {
    float sign_ = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + normal.z);
    float b = normal.x * normal.y * a;
    tangent = vec3(1.0 + sign_ * normal.x * normal.x * a, sign_ * b, -sign_ * normal.x);
    bitangent = vec3(b, sign_ + normal.y * normal.y * a, -normal.y);
}

// Sampling the GGX Distribution of Visible Normals (Heitz 2018). `Ve` is the view direction in
// tangent space (Z = macro-surface normal); returns the sampled half-vector H, also in tangent
// space. The caller reconstructs the world-space half vector via the same tangent/bitangent basis
// used to build `Ve`, then the reflected sample direction via `reflect(-V, H)`.
vec3 SampleGGXVNDF(vec2 xi, vec3 Ve, float roughness) {
    // Section 3.2: transform the view direction into the hemisphere configuration.
    vec3 Vh = normalize(vec3(roughness * Ve.x, roughness * Ve.y, Ve.z));

    // Section 4.1: orthonormal basis in the hemisphere configuration.
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = (lensq > 0.0) ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);

    // Section 4.2: parameterization of the projected area.
    float r = sqrt(xi.x);
    float phi = TAU * xi.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    // Section 4.3: reprojection onto the hemisphere.
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Section 3.4: transform back to the ellipsoid configuration -- this IS the sampled half-vector.
    return normalize(vec3(roughness * Nh.x, roughness * Nh.y, max(0.0, Nh.z)));
}

#endif // GGX_BRDF_GLSL
