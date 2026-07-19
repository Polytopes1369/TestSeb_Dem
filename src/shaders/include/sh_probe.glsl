#ifndef SH_PROBE_GLSL
#define SH_PROBE_GLSL

#include "include/math_utils.glsl"

// Shared L1 spherical-harmonics probe helpers for renderer::ScreenProbeGIPass (F9, UE5.8 parity
// roadmap: restored after an earlier perf pass deleted it outright, once replaced here by
// renderer::ScreenTracePass's own full-resolution march -- see ScreenProbeGIPass.h's own "F9" class
// comment note). L1 SH (4 coefficients per color channel) was chosen over a raw per-ray octahedral
// radiance atlas because it made both the gather pass' bilateral 4-probe blend and the temporal
// pass' exponential blend a trivial per-coefficient mix(), with no resampling between mismatched
// octahedral grids. Included by ScreenProbeTrace.comp (SHBasis4/FibonacciSphereDirection/
// JitterDirection/AccumulateSHSample) and ScreenProbeGather.comp (EvaluateIrradianceSH).
//
// Coefficient order (matches the 4 channels of one rgba16f probe atlas texel, one texel per color
// channel): c0 = Y00 (constant term), c1 = Y1,-1, c2 = Y1,0, c3 = Y1,1 -- the standard real SH
// basis (Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance Environment Maps").

const float kSH_Y00 = 0.282095;
const float kSH_Y1 = 0.488603; // Shared normalization constant for Y1,-1 / Y1,0 / Y1,1.

// Ramamoorthi & Hanrahan's cosine-lobe convolution coefficients for SH bands 0 and 1 -- turn a
// raw radiance-signal SH projection into the SH projection of the COSINE-WEIGHTED HEMISPHERE
// INTEGRAL (i.e. diffuse irradiance) at evaluation time, without ever needing to know the
// receiving surface's normal at projection time (trace time).
const float kSH_A0 = PI;
const float kSH_A1 = TAU / 3.0;

vec4 SHBasis4(vec3 d) {
    return vec4(kSH_Y00, kSH_Y1 * d.y, kSH_Y1 * d.z, kSH_Y1 * d.x);
}

// Fibonacci sphere distribution: kProbeRayCount points quasi-uniformly covering the FULL sphere
// (not just a hemisphere -- renderer::ScreenProbeGIPass's own probes are placed exactly on a
// visible surface, so roughly half of every probe's own rays self-occlude against its own local
// geometry on the very first sphere-trace/ray-query step; this is the same trade-off DDGI-style
// volumetric probes accept in exchange for the simpler, receiver-normal-independent ray set that
// pass' own task spec calls for).
vec3 FibonacciSphereDirection(uint i, uint n) {
    float y = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float radius = sqrt(max(0.0, 1.0 - y * y));
    float theta = GOLDEN_ANGLE * float(i);
    return vec3(cos(theta) * radius, y, sin(theta) * radius);
}

// Small per-frame rotation (two hash-derived angles, applied as two axis rotations) so
// consecutive frames' 64-ray sets sample different sub-directions of the sphere -- free extra
// angular coverage once combined with the temporal pass' exponential accumulation. Not a
// uniformly-distributed random SO(3) rotation (unnecessary rigor for a decorrelation jitter).
vec3 JitterDirection(vec3 d, uint frameIndex) {
    float yaw = hashFloat(uint64_t(frameIndex), 0x9E3779B9UL) * TAU;
    float pitch = hashFloat(uint64_t(frameIndex), 0x517CC1B7UL) * TAU;

    float cy = cos(yaw), sy = sin(yaw);
    vec3 d1 = vec3(d.x * cy - d.z * sy, d.y, d.x * sy + d.z * cy);

    float cp = cos(pitch), sp = sin(pitch);
    return vec3(d1.x, d1.y * cp - d1.z * sp, d1.y * sp + d1.z * cp);
}

// Accumulates one (direction, radiance) sample's contribution into the running SH sums. `weight`
// is the Monte Carlo integration weight for N uniform full-sphere samples (4*PI/N -- see
// ProjectSphereSamplesToSH's own comment); passed in rather than baked in so a caller could
// re-weight individual samples if it ever needed to (this codebase's own callers always pass the
// same constant weight for every one of the 64 rays).
void AccumulateSHSample(vec3 dir, vec3 radiance, float weight, inout vec4 shR, inout vec4 shG, inout vec4 shB) {
    vec4 basis = SHBasis4(dir) * weight;
    shR += radiance.r * basis;
    shG += radiance.g * basis;
    shB += radiance.b * basis;
}

// Evaluates the cosine-weighted irradiance at `normal` from a probe's stored raw-radiance SH1
// projection (see AccumulateSHSample) -- applies the Ramamoorthi/Hanrahan cosine-lobe convolution
// coefficients (kSH_A0/kSH_A1) at evaluation time, so the stored coefficients stay a normal-
// independent representation of the probe's full surrounding radiance field.
vec3 EvaluateIrradianceSH(vec4 shR, vec4 shG, vec4 shB, vec3 normal) {
    vec4 basis = SHBasis4(normal);
    vec4 convolved = vec4(kSH_A0 * basis.x, kSH_A1 * basis.yzw);
    return vec3(dot(shR, convolved), dot(shG, convolved), dot(shB, convolved));
}

#endif
