#version 460
#extension GL_GOOGLE_include_directive : enable

// Particle system Subtask 4 (particle_system_integration_plan.md, project root): draws every alive
// particle as a camera-facing billboard quad with NO bound vertex/index buffer -- gl_VertexIndex
// (0-5, two triangles) generates the quad's UV corners via a fixed lookup table, mirroring this
// codebase's own established "no vertex buffer" idiom (see src/shaders/src/Debug/DebugText.vert's
// identical kQuadCorners table). gl_InstanceIndex selects which of this frame's SORTED alive
// particles this instance draws -- Subtask 3's ParticleSort.comp has already written
// SortedPairsBuffer[0..aliveCount) in back-to-front order and renderer::ParticleSystemPass::
// RecordSort has already copied aliveCount into the indirect-draw buffer's own instanceCount field,
// so vkCmdDrawIndirect (this codebase's first user of the plain, non-indexed indirect draw command)
// only ever instantiates real, alive, correctly-ordered particles.

#include "include/ParticleCommon.glsl"

struct SortedPair {
    uint index;
    float key;
};
layout(std430, set = 1, binding = 0) readonly buffer SortedPairsBuffer {
    SortedPair sortedPairs[];
};

// std140 mirror of renderer::ParticleSystemPass.cpp's own (anonymous-namespace) ParticleRenderParamsUBO.
layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj; // Fragment-stage only (soft-particle scene reconstruction) -- declared here too since std140 blocks must match byte-for-byte across every stage that binds them.
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
} g_Params;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec4 outColor;
layout(location = 3) out float outNormalizedAge;
layout(location = 4) flat out uint outKind; // kParticleKindEmber/Rain/Snow (ParticleCommon.glsl) -- ParticleRender.frag branches its sprite-shape mask on this.
// Niagara-parity roadmap, subtask B3 (sprite orientation & procedural sub-variation): per-particle
// inputs ParticleRender.frag needs to perturb its analytic soft-circle mask -- see this file's own
// main() comment and ParticleRender.frag's own shapeMask comment for the full contract. Both are
// uniform across one particle's whole quad (every corner gets the SAME value, exactly like outKind
// above) -- outSubVariationSeed must be `flat` (GLSL requires this for any integer varying);
// outSubVariationStrength does not strictly need it (linearly interpolating an equal value at every
// vertex trivially reproduces that same value at every fragment), but is still one constant per
// particle, not a true per-vertex gradient.
layout(location = 5) flat out uint outSubVariationSeed;
layout(location = 6) out float outSubVariationStrength;

const vec2 kQuadCorners[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));

void main() {
    uint particleIndex = sortedPairs[gl_InstanceIndex].index;
    Particle p = particles[particleIndex];
    uint kind = UnpackParticleKind(p.randomSeed);

    // Niagara-parity roadmap (bundled B1/B2/B3 render-mode workstream): this billboard pipeline must
    // now SKIP any kKindEmber particle whose own emitter opted into Mesh Particle (B1) or
    // Ribbon/Trail (B2) render mode -- ParticleMeshRender.vert/ParticleRibbonRender.vert draw those
    // instead, this same frame, against the SAME sorted alive-particle list (see either shader's own
    // header comment). Precipitation (rain/snow) always bypasses this check and renders as its
    // original billboard streak/flake regardless -- it has no EmitterParams slot of its own
    // (Particle::emitterIndex defaults to 0 for it, see that field's own comment), so its kind alone
    // (not kParticleKindEmber) is what exempts it here. Every skipped instance is pushed to a
    // degenerate, always-clipped position -- same "skip via clip-space discard, not a separate
    // compaction pass" convention ParticleMeshRender.vert's own gating uses, for the identical
    // "avoid touching RecordSort()/ParticleSort.comp" reason.
    bool isBillboardCandidate = (kind != kParticleKindEmber) || (emitters[p.emitterIndex].renderMode == 0u);
    if (!isBillboardCandidate) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outWorldPos = vec3(0.0);
        outUV = vec2(0.0);
        outColor = vec4(0.0);
        outNormalizedAge = 0.0;
        outKind = kind;
        outSubVariationSeed = 0u;
        outSubVariationStrength = 0.0;
        return;
    }

    // Only meaningful for kKindEmber particles (precipitation has no real EmitterParams slot of its
    // own -- p.emitterIndex is an unused default 0 for it, see that field's own comment -- so this
    // read is a harmless throwaway for rain/snow, exactly like ParticleMeshRender.vert's own
    // identical pattern).
    EmitterParams e = emitters[p.emitterIndex];

    vec2 corner = kQuadCorners[gl_VertexIndex];
    vec2 offset = corner - 0.5;

    // Billboard-plane orientation. Three cases now (Niagara-parity roadmap, subtask B3 added the
    // 3rd):
    //   1. Rain (kParticleKindRain): unchanged from before this roadmap step -- aligns the quad's
    //      LOCAL +Y axis with its OWN velocity projected onto the camera's right/up plane, stretching
    //      a thin streak along the direction it is actually falling/drifting (see ParticleSimulation.
    //      comp's own SpawnPrecipitationParticle comment for why rain's `size` is deliberately
    //      thin-and-long rather than square).
    //   2. Ember with spriteOrientationMode == 1 (velocity-aligned, B3's new option): reuses the
    //      EXACT SAME velocity-onto-camera-plane projection as rain above, just gated on the
    //      emitter's own live-tunable choice instead of being hardcoded to one particle kind --
    //      lets e.g. a spark/debris emitter visibly orient along its own flight direction instead of
    //      always facing the camera at a fixed spin.
    //   3. Everything else (embers/snow with spriteOrientationMode == 0, the default): unchanged
    //      Subtask 4 original behavior -- rotate the quad-corner offset by the particle's own stored
    //      `rotation` (a fixed per-particle spin looks right for a roughly-round spark or snowflake
    //      sprite that doesn't need to visibly track its own motion).
    // This is the same rotatedOffset formula in every case, just sourcing cosR/sinR from a different
    // angle.
    bool velocityAligned = (kind == kParticleKindRain) || (kind == kParticleKindEmber && e.spriteOrientationMode == 1u);
    float cosR, sinR;
    if (velocityAligned) {
        vec2 velCameraSpace = vec2(dot(p.velocity, g_Params.cameraRight), dot(p.velocity, g_Params.cameraUp));
        float velLen = length(velCameraSpace);
        vec2 streakDir = velLen > 1.0e-4 ? (velCameraSpace / velLen) : vec2(0.0, 1.0); // Default to straight-down streak if the camera happens to be looking exactly along the velocity vector.
        // streakDir IS (sin(theta), cos(theta)) for the angle theta between +Y (cameraUp) and the
        // velocity direction -- identical algebraic role to cos(p.rotation)/sin(p.rotation) below.
        sinR = streakDir.x;
        cosR = streakDir.y;
    } else {
        cosR = cos(p.rotation);
        sinR = sin(p.rotation);
    }
    vec2 rotatedOffset = vec2(offset.x * cosR - offset.y * sinR, offset.x * sinR + offset.y * cosR);

    vec3 worldPos = p.position
        + rotatedOffset.x * p.size.x * g_Params.cameraRight
        + rotatedOffset.y * p.size.y * g_Params.cameraUp;

    outWorldPos = worldPos;
    outUV = corner;
    outColor = p.color;
    // Remaining-life fraction in [0,1] -- ParticleRender.frag's own lifeFade reads this to fade the
    // sprite out near the end of its life instead of vanishing the instant `life` crosses 0 (this
    // particle would already have been recycled to the dead-list by then anyway, by
    // ParticleSimulation.comp's own Update pass -- this is purely a visual approach-to-zero fade).
    outNormalizedAge = clamp(p.life / max(p.maxLife, 1.0e-5), 0.0, 1.0);
    outKind = kind;
    // Niagara-parity roadmap, subtask B3 (procedural sub-variation): only meaningful for embers --
    // snow reuses the same soft-circle mask branch in ParticleRender.frag, but must NOT pick up
    // EMITTERS[0]'s (or whichever slot p.emitterIndex happens to equal) subVariationStrength, since
    // precipitation has no real emitter of its own (see this function's own `e` comment above).
    // p.randomSeed's low bits are already the per-particle PRNG state SpawnParticle re-seeded at
    // spawn (see that field's own declaration comment in ParticleCommon.glsl) -- reused here as a
    // stable per-particle hash source rather than introducing a second seed field.
    outSubVariationSeed = p.randomSeed;
    outSubVariationStrength = (kind == kParticleKindEmber) ? e.subVariationStrength : 0.0;

    gl_Position = g_Params.viewProj * vec4(worldPos, 1.0);
}
