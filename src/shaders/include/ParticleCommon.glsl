#ifndef PARTICLE_COMMON_GLSL
#define PARTICLE_COMMON_GLSL

// Shared GPU-driven particle system declarations (Subtask 1 of the particle system integration
// plan, see particle_system_integration_plan.md at the project root). Every particle-system shader
// (ParticleSimulation.comp, ParticleSort.comp, ParticleRender.vert/.frag -- Subtasks 2-5) includes
// this file and binds set 0 exactly as declared below: unlike shadow_page_table.glsl/
// shadow_sun_sampling.glsl (whose binding INDICES are #define-redirected by each includer, since
// those buffers sit at different binding slots inside different consumer shaders' own descriptor
// sets), every particle shader owns a dedicated, identically-laid-out descriptor set 0 --
// renderer::ParticleSystemPass builds exactly one VkDescriptorSetLayout (see its own class comment)
// and binds it unmodified for every stage's pipeline, so hardcoding the 4 bindings here is correct
// and avoids the indirection those other shaders need.

// Mirrors renderer::GpuParticle (src/renderer/passes/ParticleSystemPass.h) byte-for-byte -- 80
// bytes, std430. vec3 members are 16-byte aligned in std430, so the trailing scalar after each vec3
// packs into the same 16-byte slot with no manual padding required (position+life,
// velocity+maxLife); color is a plain vec4 (16 bytes); size+rotation+randomSeed closes that slot
// exactly (8 + 4 + 4); emitterIndex (multi-emitter roadmap, subtask A1) closes a final dedicated
// 16-byte slot with 3 reserved pad floats for future per-particle authoring data (e.g. a future
// color/size-over-life curve's sample parameter).
struct Particle {
    vec3 position;
    float life;        // Seconds remaining before this particle dies (Subtask 2 counts this down and recycles the slot into DeadListBuffer at <= 0).
    vec3 velocity;
    float maxLife;      // Seconds this particle was spawned with -- (maxLife - life) / maxLife gives a normalized age in [0,1] for fade/size curves.
    vec4 color;
    vec2 size;
    float rotation;      // Radians, billboard-plane rotation (Subtask 4).
    uint randomSeed;     // Per-particle PRNG state, re-seeded at spawn (Subtask 2). Precipitation
                          // feature: the TOP 2 BITS (bits 30-31, see PackParticleKind/UnpackParticleKind
                          // below) are stolen to tag which emitter "kind" spawned this particle --
                          // randomSeed is otherwise dead after spawn (SpawnParticle re-seeds it once, no
                          // other consumer -- CPU or GPU -- ever reads it as a raw PRNG value again), so
                          // this is where the tag lives even though the multi-emitter roadmap (subtask
                          // A1, below) later added spare pad floats to this struct for other reasons --
                          // moving the tag would mean touching every already-tested precipitation
                          // call site for zero functional benefit. The remaining 30 bits keep plenty of
                          // entropy for UpdateParticle's own per-particle wobble phase (Subtask
                          // precipitation, snow horizontal drift).
    // Multi-emitter roadmap (subtask A1): which EmitterParamsBuffer slot spawned this particle -- only
    // meaningful for kKindEmber particles (see below); precipitation's own physics (mode == 2 spawn,
    // the kind-branch in UpdateParticle) never reads this field, it uses randomSeed's packed kind tag
    // and the separate PrecipitationParamsUBO instead.
    uint emitterIndex;
    // Subtask A4 (color-over-life / size-over-life curves): this particle's own spawn-time BASE size
    // -- the mix(sizeMin, sizeMax, random) roll SpawnParticle drew -- preserved here, separate from
    // `size` above, because UpdateParticle now overwrites `size` every single frame with
    // `baseSize * SampleSizeCurve(age)`. Without a separately stored base, next frame's multiply would
    // compound against an ALREADY-curve-modulated value instead of the original per-particle roll,
    // drifting the size every frame instead of following the curve. Mirrors renderer::GpuParticle::
    // baseSize (src/renderer/passes/ParticleSystemPass.h) byte-for-byte -- repurposes what was `_pad0`
    // there (this struct's total size does not change). Only meaningful for kKindEmber particles --
    // precipitation's own rain/snow sizes are asymmetric width/length (see this file's own
    // SpawnPrecipitationParticle comment in ParticleSimulation.comp) and never reach the curve-
    // evaluation code path at all (see UpdateParticle's own comment on why that branch is ember-only).
    float baseSize;
    float _pad1, _pad2;
};

// Per-emitter, live-tunable spawn/physics parameters -- one instance per active emitter slot (see
// renderer::ParticleSystemPass::kMaxEmitters), re-uploaded in full every RecordSimulate() call since
// every field is directly editable via main.cpp's Particles ImGui tab
// (config::particles::EMITTERS[]). Mirrors renderer::ParticleSystemPass::EmitterParams byte-for-byte
// -- 224 bytes, std430, same "flat floats, vec3 packs with trailing scalar" layout as Particle above.
// Only consumed for kKindEmber particles -- precipitation (rain/snow) has its own dedicated
// PrecipitationParamsUBO (this file's own binding declarations further down) instead.
//
// Module stack roadmap (subtask A3): adds two independently-toggleable force modules on top of the
// existing gravity/wind-drag/SDF-bounce physics below (which are unchanged) -- see
// ParticleSimulation.comp's UpdateParticle for where these are applied, additively, in the ember
// (non-precipitation) branch only. A full visual-scripting module graph is out of scope for this
// project (see renderer::ParticleSystemPass::EmitterParams' own header comment); this is instead a
// small fixed-size data-driven set, matching this codebase's "no data in the .exe" discipline.
struct EmitterParams {
    vec3 position;
    float shapeParam0;      // Spawn-shape parameter: Sphere (spawnShape==1) = radius in world units; Cone (spawnShape==0) = unused.
    vec4 color;             // Base spawn color (Subtask A4 will add color-over-life curves on top of this).
    float sizeMin, sizeMax;
    float lifetimeMin, lifetimeMax;
    float gravityY;          // World-space Y acceleration, m/s^2 (replaces the old single global config::particles::GRAVITY).
    float bounceElasticity;  // [0,1] -- fraction of normal-relative speed kept after a Global SDF collision.
    float friction;          // [0,1] -- fraction of tangential-relative speed kept after a Global SDF collision.
    float dragCoefficient;   // How strongly velocity relaxes toward the local Atmos wind vector each second.
    uint spawnShape;         // 0 = Cone burst (legacy "embers" launch direction/jitter), 1 = Sphere volume drift spawn.
    // Module stack roadmap (subtask A3): curl-noise turbulence module -- replaces the 3 previously
    // unused trailing pad floats that used to close spawnShape's own 16-byte slot (no size change).
    uint curlNoiseEnabled;    // Nonzero = apply a divergence-free curl-noise force to velocity every UpdateParticle() call.
    float curlNoiseStrength;  // Force magnitude, m/s^2 applied to velocity per second at full strength.
    float curlNoiseScale;     // World-space frequency multiplier fed into the curl-noise field (bigger = finer swirls).
    // Module stack roadmap (subtask A3): radial attractor/repulsor module -- its own point is an
    // OFFSET from this emitter's own live `position` (tags along with a moving emitter), not a fixed
    // world point -- own new 16+16-byte slot pair, same vec3+trailing-scalar convention as above.
    vec3 attractorOffset;
    float attractorStrength; // Positive = attract toward the point, negative = repel away from it, m/s^2 at zero distance (before falloff below).
    float attractorRadius;   // World units -- force falls off smoothly (smoothstep) to zero at this distance.
    uint attractorEnabled;
    float _pad0, _pad1;
    // Subtask A4 (color-over-life / size-over-life curves): 4 evenly-spaced keyframes at normalized
    // age 0.0/0.33/0.67/1.0 -- UpdateParticle (ParticleSimulation.comp) linearly interpolates between
    // the two bracketing keys every frame from the particle's own (maxLife - life) / maxLife, instead
    // of SpawnParticle picking one fixed color/size for the particle's entire life. Mirrors renderer::
    // ParticleSystemPass::EmitterParams::colorCurve/sizeCurve (src/renderer/passes/
    // ParticleSystemPass.h) byte-for-byte -- see that field's own declaration comment for the full
    // "colorCurve is direct/authoritative, sizeCurve is a multiplier on sizeMin/sizeMax's own
    // per-particle roll" rationale. Tightly packed under std430 (this SSBO's own layout qualifier,
    // see this file's own header comment) -- `vec4 colorCurve[4]` is 64 bytes with no slack (vec4 is
    // already 16-byte aligned/sized), and `float sizeCurve[4]` is 16 bytes, NOT expanded to 16
    // bytes-per-element the way std140 would: std430 does not round a scalar array's stride up to a
    // vec4 multiple.
    vec4 colorCurve[4];
    float sizeCurve[4];

    // Niagara-parity roadmap (bundled B1 "Mesh Particle" + B2 "Ribbon/Trail" + B3 "sprite
    // orientation/sub-variation" workstream): ONE render-mode enum shared by all three subtasks --
    // mirrors renderer::ParticleSystemPass::EmitterParams::renderMode/meshArchetype/ribbonWidth/
    // spriteOrientationMode/subVariationStrength (src/renderer/passes/ParticleSystemPass.h)
    // byte-for-byte -- see that field's own declaration comment for the full contract. Appended at
    // the very END of this struct for the same "minimize textual overlap with other parallel
    // workstreams" reason as that file's own comment. renderMode == 0 (Billboard) is the default for
    // every field below, so an emitter that never touches any of these fields renders EXACTLY as it
    // did before this roadmap step -- see ParticleRender.vert/.frag's own gating-check comments.
    uint renderMode;            // 0 = Billboard (default), 1 = Mesh Particle (B1), 2 = Ribbon/Trail (B2). Only meaningful for kKindEmber particles.
    uint meshArchetype;         // Render mode 1 only -- 0 = box, 1 = icosphere (see ParticleMeshRender.vert's own comment).
    float ribbonWidth;          // Render mode 2 only -- half-width, world units, of the trail's cross-section quad-strip.
    uint spriteOrientationMode; // Render mode 0 only -- 0 = camera-facing (original default), 1 = velocity-aligned (B3).
    float subVariationStrength; // Render mode 0 only -- [0,1], B3's procedural per-particle shape perturbation strength (0.0 = pixel-identical to pre-B3).
    float _pad2, _pad3, _pad4;
};

// Particle "kind" tag, packed into Particle.randomSeed's top 2 bits (see that field's own comment).
// kKindEmber is 0 so every particle spawned before this feature existed (and every non-precipitation
// spawn today) is correctly tagged without needing to touch SpawnParticle's existing embers path.
const uint kParticleKindEmber = 0u;
const uint kParticleKindRain = 1u;
const uint kParticleKindSnow = 2u;

uint PackParticleSeed(uint kind, uint rawSeed) {
    return (kind << 30u) | (rawSeed & 0x3FFFFFFFu);
}
uint UnpackParticleKind(uint packedSeed) {
    return packedSeed >> 30u;
}
uint UnpackParticleRawSeed(uint packedSeed) {
    return packedSeed & 0x3FFFFFFFu;
}

// Double-buffered particle state (renderer::ParticleSystemPass::m_ParticleBuffer[2]) -- Subtask 2's
// simulation compute shader reads the PREVIOUS frame's buffer and writes the buffer for THIS frame,
// then the two swap for the next dispatch (ping-pong, same convention as e.g.
// renderer::ReflectionPass's own temporal history buffers), so every particle shader binds set 0
// against whichever GpuBuffer is "current" for that call -- there is exactly one binding 0 in this
// layout, not two, because the two physical buffers are never both bound at once (unlike a
// read-this/write-that image pair, the double buffer here exists purely to let last frame's data
// keep being READ by late-frame consumers -- e.g. Subtask 4's render pass -- while this frame's
// simulation dispatch is already writing the other one).
layout(std430, set = 0, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

// Free-list of particle-buffer slot indices not currently alive. Subtask 2's spawn step pops
// indices from the tail (atomicAdd(deadCount, -1) then read deadIndices[deadCount]) to initialize
// new particles; a particle's death pushes its index back (atomicAdd(deadCount, 1) then write
// deadIndices[deadCount]). Sized to kMaxParticles (renderer::ParticleSystemPass.h) and initialized
// at Init() time to hold every index 0..kMaxParticles-1 (see ParticleSystemPass::Init's own "seed
// the dead-list" comment) so every slot starts dead/available.
layout(std430, set = 0, binding = 1) buffer DeadListBuffer {
    uint deadIndices[];
};

// Compacted list of currently-alive particle-buffer slot indices, rebuilt every frame by Subtask
// 2's simulation dispatch (a live particle appends its own index via
// atomicAdd(aliveCount, 1)). Subtask 3's sort pass reads this to know which particles to key/sort,
// and Subtask 4's render pass indexes this (after sorting) via gl_VertexIndex / gl_InstanceIndex to
// know which Particle to billboard -- never iterates the raw ParticleBuffer directly, since most of
// its kMaxParticles slots are typically dead.
layout(std430, set = 0, binding = 2) buffer AliveListBuffer {
    uint aliveIndices[];
};

// Single 16-byte counter block, one instance for the whole system (not per-particle). `deadCount`/
// `aliveCount` are the CURRENT lengths of the two lists above (both mutated via atomics from
// Subtask 2's compute shader, one thread per particle slot); `spawnQueue` is how many NEW particles
// this frame's simulation dispatch should attempt to spawn (written by the CPU/caller before
// RecordSimulate -- see renderer::ParticleSystemPass::RecordSimulate's own comment -- then drained
// back toward 0 by the shader as it successfully spawns each one, blocked once deadCount reaches
// 0). `_pad0` keeps the struct's std430 size a clean 16 bytes -- no functional use.
layout(std430, set = 0, binding = 3) buffer CounterBuffer {
    uint deadCount;
    uint aliveCount;
    uint spawnQueue;
    uint _pad0;
};

// Fixed-size array of kMaxEmitters EmitterParams (renderer::ParticleSystemPass::kMaxEmitters),
// read-only from every particle shader's point of view -- the CPU side (RecordSimulate) is the only
// writer, via a full vkCmdUpdateBuffer every call (see that method's own comment). Subtask A1
// (multi-emitter roadmap): SpawnParticle indexes this by the emitter it is spawning for;
// UpdateParticle indexes this by the particle's OWN stored Particle.emitterIndex so each particle's
// physics response (gravity/bounce/friction/drag) stays correct for its emitter even if that
// emitter's live ImGui values change mid-flight.
layout(std430, set = 0, binding = 4) readonly buffer EmitterParamsBuffer {
    EmitterParams emitters[];
};

// Debug/test instrumentation only (multi-emitter roadmap, subtask A1's own validation step): one
// alive-particle counter per emitter slot, reset to 0 every RecordSimulate() call (vkCmdFillBuffer)
// and incremented by UpdateParticle() ONLY in Debug builds (see that function's own `#ifdef _DEBUG`
// guard) -- per CLAUDE.md's build-separation rule, no atomic-increment instruction is ever emitted
// into the Release SPIR-V, so this buffer's contents are simply never written to in Release (its
// allocation is the only thing that survives unconditionally, same harmless-always-present
// convention as CounterBuffer's own unused `spawnQueue`/`_pad0` fields above). Read back by
// renderer::ParticleSystemPass's own Debug-only GetLastPerEmitterAliveCountApprox(), surfaced in
// main.cpp's Particles ImGui tab so a developer can visually confirm each emitter is independently
// alive/producing particles.
layout(std430, set = 0, binding = 5) buffer PerEmitterAliveCountBuffer {
    uint perEmitterAliveCount[];
};

#endif
