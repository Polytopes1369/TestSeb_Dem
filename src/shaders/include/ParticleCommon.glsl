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
    uint randomSeed;     // Per-particle PRNG state, re-seeded at spawn (Subtask 2).
    uint emitterIndex;   // Which EmitterParamsBuffer slot spawned this particle (subtask A1: multi-emitter).
    float _pad0, _pad1, _pad2;
};

// Per-emitter, live-tunable spawn/physics parameters -- one instance per active emitter slot (see
// renderer::ParticleSystemPass::kMaxEmitters), re-uploaded in full every RecordSimulate() call since
// every field is directly editable via main.cpp's Particles ImGui tab
// (config::particles::EMITTERS[]). Mirrors renderer::ParticleSystemPass::EmitterParams byte-for-byte
// -- 80 bytes, std430, same "flat floats, vec3 packs with trailing scalar" layout as Particle above.
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
    float _pad0, _pad1, _pad2;
};

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
