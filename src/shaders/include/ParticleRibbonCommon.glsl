#ifndef PARTICLE_RIBBON_COMMON_GLSL
#define PARTICLE_RIBBON_COMMON_GLSL

// Niagara-parity roadmap, subtask B2 (Ribbon/Trail render mode) -- per-particle-SLOT world-position
// history ring buffer, written every UpdateParticle() call (ParticleSimulation.comp) and read by
// ParticleRibbonRender.vert to build each particle's own trailing quad-strip.
//
// Deliberately a SEPARATE descriptor set/pair of buffers from ParticleCommon.glsl's own set 0 (NOT
// 2 more bindings appended to that already-widely-shared descriptor set layout) -- this way only the
// 2 shaders that actually touch ribbon history (ParticleSimulation.comp writes, ParticleRibbonRender.
// vert reads) need to bind it at all; every OTHER particle pipeline (ParticleSort.comp,
// ParticleRender.vert/.frag, ParticleMeshRender.vert/.frag) keeps its existing pipeline layout
// entirely unchanged, with zero risk of colliding with a set/binding index some OTHER parallel
// workstream might simultaneously be adding elsewhere in this same struct/descriptor-set family.
//
// Set/binding indices are #define-configurable (mirroring this codebase's own established
// shadow_page_table.glsl/world_probe_sampling.glsl convention, see e.g. ParticleRender.frag's own
// SHADOW_PAGE_TABLE_SET/WORLD_PROBE_GRID_SET defines) since the 2 consumer shaders bind this set at
// DIFFERENT indices within their own, otherwise-unrelated pipeline layouts: ParticleSimulation.comp
// puts it at set 2 (after its own set 0 particle-state / set 1 environment sets), ParticleRibbonRender.
// vert puts it at set 4 (after reusing the billboard pipeline's own 4 sets: particle-state / sort /
// render-params / lighting).
#ifndef RIBBON_HISTORY_SET
#define RIBBON_HISTORY_SET 2
#endif
#ifndef RIBBON_HISTORY_BINDING
#define RIBBON_HISTORY_BINDING 0
#endif
#ifndef RIBBON_SAMPLE_COUNT_BINDING
#define RIBBON_SAMPLE_COUNT_BINDING 1
#endif

// Ring buffer sizing: kRibbonHistorySamples entries per particle SLOT (not per emitter/kind), laid
// out contiguously (particleSlotIndex * kRibbonHistorySamples + ringSlot) -- a flat SSBO array
// rather than an array-of-structs-of-arrays, matching this codebase's own established "flat backing
// array, index arithmetic in the shader" convention for anything indexed by particle slot (see e.g.
// ParticleCommon.glsl's own DeadListBuffer/AliveListBuffer). Must match renderer::ParticleSystemPass::
// kRibbonHistorySamples exactly (that constant sizes the backing buffers this mirrors). Within the
// 4-8 sample range this roadmap step's own task description asks for -- at kMaxParticles == 65536,
// 6 samples * 16 bytes (vec4, see RibbonHistoryBuffer's own comment) * 65536 slots == 6 MiB, a
// deliberately modest budget given how many particle slots exist, not a "trail quality" ceiling a
// demoscene emitter would ever need more of.
const uint kRibbonHistorySamples = 6u;

// Position-only (vec4, .w unused -- kept vec4 rather than vec3 purely so std430 indexing is a
// trivial stride-16 multiply with no vec3-in-an-array padding surprise to reason about, matching
// this codebase's own "flat float array, no vec3-in-array" convention elsewhere).
layout(std430, set = RIBBON_HISTORY_SET, binding = RIBBON_HISTORY_BINDING) buffer RibbonHistoryBuffer {
    vec4 ribbonHistory[];
};

// One "total pushes so far" counter per particle slot (NOT pre-wrapped -- the ring index is derived
// via `% kRibbonHistorySamples` at every read/write site instead, so a reader can always tell "how
// many of my last kRibbonHistorySamples pushes are actually valid" via
// min(thisValue, kRibbonHistorySamples), correctly reporting fewer valid samples for a particle that
// has not lived long enough yet to fill the whole ring). Reset to 1 at spawn (see
// ParticleSimulation.comp's own SpawnParticle/SpawnPrecipitationParticle) -- along with writing the
// spawn position into ring slot 0 -- so a reused particle slot never inherits its PREVIOUS
// occupant's stale ring contents/count, and a fresh particle's ribbon has continuity from its own
// spawn point forward from frame 1 onward.
layout(std430, set = RIBBON_HISTORY_SET, binding = RIBBON_SAMPLE_COUNT_BINDING) buffer RibbonSampleCountBuffer {
    uint ribbonSampleCount[];
};

#endif // PARTICLE_RIBBON_COMMON_GLSL
