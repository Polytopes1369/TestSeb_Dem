#ifndef VEGETATION_COMMON_GLSL
#define VEGETATION_COMMON_GLSL

// Shared declarations for the GPU-instanced procedural vegetation scatter (UE5.8 rendering-parity
// gap G2). Unlike the just-merged renderer::ProceduralTreePass (each tree baked as a full,
// individual Nanite entity with its own cluster DAG -- correct for a handful of hero trees, but it
// does NOT scale to the hundreds/thousands of grass/shrub/rock instances a PCG-style scatter needs),
// this system follows renderer::ParticleSystemPass's structural template instead: one GPU-resident
// per-instance buffer, a GPU-driven cull compaction step, and an indirect instanced draw sharing a
// few small base meshes. See renderer::VegetationScatterPass for the C++ side.
//
// Every field/constant below is mirrored byte-for-byte on the C++ side -- keep the two in sync.

// Archetype identifiers. Each archetype is one small procedurally-generated base mesh, instanced
// many times across the terrain. Grass is a cheap crossed-card clump (billboard-style, like
// ProceduralTreePass's own leaf cards -- reused as reference); bush and rock are noise-perturbed
// low-poly blobs (see geom_scatter_grass.comp / geom_scatter_blob.comp).
const uint kVegArchetypeGrass = 0u;
const uint kVegArchetypeBush  = 1u;
const uint kVegArchetypeRock  = 2u;
const uint kVegArchetypeCount = 3u;

// Per-instance record -- 32 bytes, std430. vec3 members are 16-byte aligned in std430, so `scale`
// packs into position's trailing slot with no manual padding; the following two uints + yaw + one
// pad float close a second 16-byte slot exactly. Mirrors renderer::VegetationScatterPass::
// GpuVegetationInstance exactly.
struct VegetationInstance {
    vec3 position;   // World-space base position (the mesh's local origin lands here).
    float scale;     // Uniform scale multiplier applied to the base mesh's local coordinates.
    float yaw;       // Rotation about world +Y, radians.
    uint archetype;  // kVegArchetypeGrass / Bush / Rock -- selects which base mesh this instance draws.
    uint tintSeed;   // Per-instance PRNG seed driving the procedural color-tint variation (see VegetationTint below).
    float _pad0;
};

// PCG-style integer hash (same avalanche-mixer family as displacement_noise.glsl's HashUint3D):
// deterministic, no state, good bit diffusion -- used both for scatter placement jitter and for the
// per-instance color/scale/yaw variation so a grass field never looks like one cloned instance.
uint VegHashU(uint x) {
    x += 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
uint VegHashCombine(uint a, uint b) { return VegHashU(a ^ (b * 0x9e3779b9u)); }
float VegHashToUnitFloat(uint h) { return float(h & 0x00FFFFFFu) / float(0x01000000u); } // [0, 1)

// Procedural per-instance color tint. Returns a multiplicative RGB tint around 1.0: a small hue
// shift plus a darkness/brightness jitter, seeded by the instance's tintSeed. Keeps a scattered
// field visually varied without any texture asset (CLAUDE.md's zero-data-asset rule), same analytic
// approach every other material in this renderer uses.
vec3 VegetationTint(uint tintSeed, uint archetype) {
    float t0 = VegHashToUnitFloat(VegHashU(tintSeed));
    float t1 = VegHashToUnitFloat(VegHashU(tintSeed ^ 0x1234567u));
    float t2 = VegHashToUnitFloat(VegHashU(tintSeed ^ 0x89abcdefu));
    // Brightness jitter in [0.75, 1.15] -- some instances read darker (in shadow of their own
    // clump), some slightly lighter.
    float brightness = 0.75 + t0 * 0.40;
    // Subtle per-channel hue drift; grass leans greener/yellower, rock/bush stay near-neutral.
    vec3 hue = (archetype == kVegArchetypeGrass)
        ? vec3(0.85 + t1 * 0.20, 1.0, 0.80 + t2 * 0.25)   // green/yellow bias
        : vec3(0.92 + t1 * 0.14, 0.94 + t2 * 0.12, 0.92 + t0 * 0.14);
    return hue * brightness;
}

#endif // VEGETATION_COMMON_GLSL
