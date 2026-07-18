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

// Mirrors renderer::GpuParticle (src/renderer/passes/ParticleSystemPass.h) byte-for-byte -- 64
// bytes, std430. vec3 members are 16-byte aligned in std430, so the trailing scalar after each vec3
// packs into the same 16-byte slot with no manual padding required (position+life,
// velocity+maxLife); color is a plain vec4 (16 bytes); size+rotation+randomSeed closes the last
// 16-byte slot exactly (8 + 4 + 4).
struct Particle {
    vec3 position;
    float life;        // Seconds remaining before this particle dies (Subtask 2 counts this down and recycles the slot into DeadListBuffer at <= 0).
    vec3 velocity;
    float maxLife;      // Seconds this particle was spawned with -- (maxLife - life) / maxLife gives a normalized age in [0,1] for fade/size curves.
    vec4 color;
    vec2 size;
    float rotation;      // Radians, billboard-plane rotation (Subtask 4).
    uint randomSeed;     // Per-particle PRNG state, re-seeded at spawn (Subtask 2).
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

#endif
