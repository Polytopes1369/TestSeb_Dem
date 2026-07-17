#ifndef ENHANCED_DISPLACEMENT_GLSL
#define ENHANCED_DISPLACEMENT_GLSL

// Phase 1 (Nanite advanced) -- "dynamic tessellation" reinterpreted the way real UE5.8 Nanite
// actually does it: there is no hardware tesc/tese stage anywhere in this engine's cluster raster
// path (ClusterHardwareRasterPass/ClusterSoftwareRasterPass both rasterize pre-baked micro-
// triangles with no tessellator slot), and real Nanite doesn't tessellate clusters either -- it
// relies on pre-baked micro-poly density plus per-vertex World Position Offset for displacement.
// This file is exactly that: a richer, higher-frequency multi-octave WPO on top of the existing
// single-sine sway in wpo_deformation.glsl, applied ADDITIVELY (not a replacement) and gated per-
// entity via ENTITY_FLAG_HAS_ENHANCED_DISPLACEMENT so only the demo entity pays the extra ALU cost.
//
// ENHANCED_DISPLACEMENT_MAX_AMPLITUDE is a provable bound, not a guess: EnhancedDisplacementFBM's
// four octave amplitudes (0.5 + 0.25 + 0.125 + 0.0625 = 0.9375) sum to strictly less than 1.0, and
// each octave's signed noise term is clamped to [-1, 1] by construction, so |FBM(p)| <= 0.9375 for
// any input -- multiplying by this constant therefore bounds ApplyEnhancedDisplacement's output to
// strictly less than this amplitude. This mirrors ClusterCullMetadata::maxWPOAmplitude's own
// contract (see wpo_deformation.glsl's header comment): the culling/LOD-error bound this feature's
// flag adds (ClusterDAGScreenError.comp, ClusterLODCompact.comp) must never be exceeded by the
// actual displacement, or displaced geometry could pop through a bounding volume the culling pass
// already decided was safely outside the frustum/occluded.
#define ENHANCED_DISPLACEMENT_MAX_AMPLITUDE 0.06

// Wang-hash-style integer mixer for a 3D cell coordinate, extending ProceduralMaskGenerate.comp's
// own HashUint(uint)/2D-cell-mixing pattern to a third axis (same style, kept consistent rather
// than inventing a new hash family) -- purely cosmetic noise, not security-sensitive.
uint EnhancedDisplacementHash3D(uint ix, uint iy, uint iz, uint seed) {
    uint x = ix * 1973u + iy * 9277u + iz * 26699u + seed * 60493u;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Trilinear value noise at hashed integer cell corners (8 corners of the enclosing unit cube),
// smoothstep-blended per axis so adjacent cells join without hard seams -- the direct 3D extension
// of ProceduralMaskGenerate.comp's ValueNoise(vec2, uint), same technique, one more axis.
// Returns a value in [0, 1].
float EnhancedDisplacementValueNoise3D(vec3 p, uint seed) {
    vec3 cell = floor(p);
    vec3 frac = p - cell;
    vec3 smoothFrac = frac * frac * (3.0 - 2.0 * frac);

    uint ix = uint(int(cell.x)), iy = uint(int(cell.y)), iz = uint(int(cell.z));

    float c000 = float(EnhancedDisplacementHash3D(ix,      iy,      iz,      seed) & 0xFFFFu) / 65535.0;
    float c100 = float(EnhancedDisplacementHash3D(ix + 1u, iy,      iz,      seed) & 0xFFFFu) / 65535.0;
    float c010 = float(EnhancedDisplacementHash3D(ix,      iy + 1u, iz,      seed) & 0xFFFFu) / 65535.0;
    float c110 = float(EnhancedDisplacementHash3D(ix + 1u, iy + 1u, iz,      seed) & 0xFFFFu) / 65535.0;
    float c001 = float(EnhancedDisplacementHash3D(ix,      iy,      iz + 1u, seed) & 0xFFFFu) / 65535.0;
    float c101 = float(EnhancedDisplacementHash3D(ix + 1u, iy,      iz + 1u, seed) & 0xFFFFu) / 65535.0;
    float c011 = float(EnhancedDisplacementHash3D(ix,      iy + 1u, iz + 1u, seed) & 0xFFFFu) / 65535.0;
    float c111 = float(EnhancedDisplacementHash3D(ix + 1u, iy + 1u, iz + 1u, seed) & 0xFFFFu) / 65535.0;

    float x00 = mix(c000, c100, smoothFrac.x);
    float x10 = mix(c010, c110, smoothFrac.x);
    float x01 = mix(c001, c101, smoothFrac.x);
    float x11 = mix(c011, c111, smoothFrac.x);

    float y0 = mix(x00, x10, smoothFrac.y);
    float y1 = mix(x01, x11, smoothFrac.y);

    return mix(y0, y1, smoothFrac.z);
}

// 4-octave fractional Brownian motion, geometric amplitude falloff (0.5/0.25/0.125/0.0625, see
// ENHANCED_DISPLACEMENT_MAX_AMPLITUDE's header comment for why this sums provably below 1.0), and a
// non-power-of-2 lacunarity (x2.17) between octaves to avoid axis-aligned repetition artifacts.
// Returns a signed value, bounded to (-0.9375, 0.9375).
float EnhancedDisplacementFBM(vec3 p, uint seed) {
    float amplitude = 0.5;
    float frequency = 1.0;
    float sum = 0.0;
    for (int octave = 0; octave < 4; ++octave) {
        float n = EnhancedDisplacementValueNoise3D(p * frequency, seed + uint(octave) * 101u);
        sum += (n * 2.0 - 1.0) * amplitude; // remap [0,1] -> [-1,1] before weighting
        amplitude *= 0.5;
        frequency *= 2.17;
    }
    return sum;
}

// worldPos: the vertex's world-space position AFTER the existing WPO sway has already been applied
// (this function's displacement composes additively on top of it, never replacing it).
// entityPivot: EntityTransform.center for this vertex's entity -- the displacement direction is
// radial from this pivot (normalize(worldPos - entityPivot)), a deliberately simple, isotropically-
// bounded stand-in for the true per-vertex surface normal: neither ClusterRaster.vert nor
// cluster_software_raster_core.glsl decode the cluster's stored normal in their vertex-position
// path (only ClusterResolve.comp's shading path does), so reusing the true normal here would add
// bandwidth to the hot vertex-pulling code path for both raster backends just for this cosmetic
// effect. Radial-from-pivot reads correctly on convex, roughly-spherical/cylindrical demo shapes.
// clusterID: this cluster's persistent geometry::ClusterIndexEntry::clusterID, mixed into the noise
// seed so neighboring clusters don't share an identical bump pattern.
// globalTime: seconds elapsed since startup (same WPOGlobalsUBO uniform wpo_deformation.glsl reads)
// -- scaled down heavily (x0.05) for a slow "breathing" drift, deliberately distinct in character
// from the sway's fast x1.5 oscillation so the two effects read as visually independent.
vec3 ApplyEnhancedDisplacement(vec3 worldPos, vec3 entityPivot, uint clusterID, float globalTime) {
    vec3 radial = worldPos - entityPivot;
    float radialLength = length(radial);
    // Guard against a vertex landing exactly on the pivot (radial would be a zero-length vector,
    // and normalize() of a zero vector is undefined in GLSL) -- extremely unlikely for a surface
    // vertex of a non-degenerate mesh, but cheap and correct to guard explicitly rather than assume.
    if (radialLength < 1e-5) {
        return worldPos;
    }
    vec3 radialDir = radial / radialLength;

    vec3 samplePos = worldPos * 4.0 + vec3(0.0, 0.0, globalTime * 0.05);
    float fbm = EnhancedDisplacementFBM(samplePos, clusterID * 47u + 11u);

    return worldPos + radialDir * (fbm * ENHANCED_DISPLACEMENT_MAX_AMPLITUDE);
}

#endif // ENHANCED_DISPLACEMENT_GLSL
