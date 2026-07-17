#ifndef DISPLACEMENT_NOISE_GLSL
#define DISPLACEMENT_NOISE_GLSL

// Phase 7a (UE5.8 parity roadmap, hero asset tessellation): 3D hashed value-noise + fbm, extending
// ProceduralMaskGenerate.comp's own 2D two-octave formula (the only procedural noise precedent in
// this codebase) to a 3rd axis and 4 octaves -- mirrors that shader's exact hash mixer and
// smoothstep-blended interpolation style/quality bar, just trilinear instead of bilinear.
// World/object-space sampling (NOT UV): the Fallback Mesh's own UVs have no seam-free guarantee
// and this codebase has no texture/material-binding system at all (see TransparentForward.vert's
// own comment on procedural_material.glsl) -- a texture-sampled heightmap would be inconsistent
// with established practice here.

// Same 3-round avalanche mixer as ProceduralMaskGenerate.comp's HashUint, applied to a 3-prime
// combine of the 3 integer cell coordinates (1973u/9277u come verbatim from that file; a 3rd,
// 42013u, extends the same mixing scheme to the new Z axis).
uint HashUint3D(uint x, uint y, uint z) {
    uint h = x * 1973u + y * 9277u + z * 42013u;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

float HashToUnitFloat(uint h) {
    return float(h & 0xFFFFu) / 65535.0;
}

// Value noise at integer cell corners, trilinearly interpolated with a smoothstep blend -- exact
// 3D extension of ProceduralMaskGenerate.comp's own ValueNoise (8 corners instead of 4).
float ValueNoise3D(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = p - cell;
    vec3 s = f * f * (3.0 - 2.0 * f);

    uint ix = uint(cell.x), iy = uint(cell.y), iz = uint(cell.z);
    float h000 = HashToUnitFloat(HashUint3D(ix,      iy,      iz));
    float h100 = HashToUnitFloat(HashUint3D(ix + 1u, iy,      iz));
    float h010 = HashToUnitFloat(HashUint3D(ix,      iy + 1u, iz));
    float h110 = HashToUnitFloat(HashUint3D(ix + 1u, iy + 1u, iz));
    float h001 = HashToUnitFloat(HashUint3D(ix,      iy,      iz + 1u));
    float h101 = HashToUnitFloat(HashUint3D(ix + 1u, iy,      iz + 1u));
    float h011 = HashToUnitFloat(HashUint3D(ix,      iy + 1u, iz + 1u));
    float h111 = HashToUnitFloat(HashUint3D(ix + 1u, iy + 1u, iz + 1u));

    float x00 = mix(h000, h100, s.x);
    float x10 = mix(h010, h110, s.x);
    float x01 = mix(h001, h101, s.x);
    float x11 = mix(h011, h111, s.x);
    float y0 = mix(x00, x10, s.y);
    float y1 = mix(x01, x11, s.y);
    return mix(y0, y1, s.z); // [0, 1)
}

const uint kDisplacementOctaves = 4u;
const float kDisplacementBaseFrequency = 2.5; // Noise cycles per world unit at octave 0.

// Fractal Brownian motion: 4 octaves, amplitude halves / frequency doubles each step, normalized
// to [-1, 1] (remapped from ValueNoise3D's own [0,1) per octave) so callers can offset a surface
// both inward and outward along its normal, not just outward.
float Fbm3D(vec3 p) {
    float amplitude = 0.5;
    float frequency = 1.0;
    float sum = 0.0;
    float maxAmplitude = 0.0;
    for (uint i = 0u; i < kDisplacementOctaves; ++i) {
        sum += (ValueNoise3D(p * frequency) * 2.0 - 1.0) * amplitude;
        maxAmplitude += amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return sum / maxAmplitude;
}

// Public entry point: samples the fbm at `worldPos * kDisplacementBaseFrequency`, scaled by
// `amplitudeScale` (world units).
float SampleDisplacement(vec3 worldPos, float amplitudeScale) {
    return Fbm3D(worldPos * kDisplacementBaseFrequency) * amplitudeScale;
}

// World units -- a small fraction of one noise cell (1/kDisplacementBaseFrequency ~= 0.4), small
// enough to approximate a derivative without under/overshooting the local curvature.
const float kCentralDifferenceEpsilon = 0.02;

// Central-difference perturbed normal: offsets `basePos` along 2 tangent directions (Duff et al.
// ONB -- a fresh, self-contained copy, same "duplicate rather than cross-include" convention
// ggx_brdf.glsl's own BuildTangentBasis already established for SurfaceCacheGIInject.comp's inline
// ONB), resamples displacement at each perturbed point, and derives the true perturbed normal from
// the cross product of the two resulting tangent-plane displacement vectors -- a genuinely correct
// finite-difference surface normal, NOT a flat-normal shortcut. At zero displacement this reduces
// exactly to `baseNormal` (cross(tangent, bitangent) == normal by the ONB's own construction).
vec3 ComputeDisplacedNormal(vec3 basePos, vec3 baseNormal, float displacementScale) {
    float sign_ = baseNormal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + baseNormal.z);
    float b = baseNormal.x * baseNormal.y * a;
    vec3 tangent = vec3(1.0 + sign_ * baseNormal.x * baseNormal.x * a, sign_ * b, -sign_ * baseNormal.x);
    vec3 bitangent = vec3(b, sign_ + baseNormal.y * baseNormal.y * a, -baseNormal.y);

    vec3 pT = basePos + tangent * kCentralDifferenceEpsilon;
    vec3 pB = basePos + bitangent * kCentralDifferenceEpsilon;

    vec3 displacedBase = basePos + baseNormal * SampleDisplacement(basePos, displacementScale);
    vec3 displacedT = pT + baseNormal * SampleDisplacement(pT, displacementScale);
    vec3 displacedB = pB + baseNormal * SampleDisplacement(pB, displacementScale);

    return normalize(cross(displacedT - displacedBase, displacedB - displacedBase));
}

#endif // DISPLACEMENT_NOISE_GLSL
