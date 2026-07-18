#version 460
#extension GL_GOOGLE_include_directive : enable

// Niagara-parity roadmap, subtask B1 (Mesh Particle render mode): draws every alive kKindEmber
// particle whose OWN emitter has EmitterParams.renderMode == 1 (Mesh Particle) as an instance of one
// of TWO small procedural meshes (box, icosphere) instead of a camera-facing billboard quad --
// renderer::ParticleSystemPass::Init() generates both meshes ONCE, at startup, by reusing
// geom_box.comp/geom_icosphere.comp (the SAME GPU generation shaders the main cluster/Nanite
// pipeline builds its own procedurally-generated entity meshes from -- see that method's own comment
// for why this is deliberately plain hardware instancing, not a virtualized cluster/DAG mesh).
//
// This pipeline reuses renderer::ParticleSystemPass's EXACT SAME 4-set pipeline layout as
// ParticleRender.vert/.frag (set 0 = particle state, set 1 = SortedPairsBuffer, set 2 =
// ParticleRenderParamsUBO, set 3 = VSM + World Probe Grid lighting) -- unlike the billboard pass,
// this one binds a REAL vertex/index buffer (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT/INDEX_BUFFER_BIT)
// rather than generating every corner from gl_VertexIndex alone, since a real 3D mesh needs real
// per-vertex position/normal data for correct silhouette + Lambertian shading (ParticleMeshRender.
// frag's own comment) -- a deliberate, documented deviation from the "no vertex buffer" convention
// the billboard/ribbon render modes use, not an oversight.
//
// This same shader module is used for BOTH archetypes (renderer::ParticleSystemPass::RecordDraw
// issues 2 separate vkCmdDrawIndexedIndirect calls, one per archetype, into disjoint sub-ranges of
// the SAME shared vertex/index buffer) -- `pc.expectedArchetype` tells this specific invocation
// which archetype it currently represents, purely for the render-mode GATING check below (the
// GEOMETRY itself is already routed correctly by the draw call's own fixed firstIndex/vertexOffset,
// this push constant is not needed for that).
//
// Render-mode gating (mirrors ParticleRender.vert's own "which render mode does THIS particle's
// emitter want" check, just for archetype 1 == Mesh Particle instead of 0 == Billboard): every
// instance whose particle does not match is pushed to a degenerate, always-clipped position instead
// of being compacted out of a separate index list -- see this shader's own main() comment for why
// (avoids touching RecordSort()/ParticleSort.comp, explicitly out of scope for this roadmap step).

#include "include/ParticleCommon.glsl"

struct SortedPair {
    uint index;
    float key;
};
layout(std430, set = 1, binding = 0) readonly buffer SortedPairsBuffer {
    SortedPair sortedPairs[];
};

// std140 mirror of renderer::ParticleSystemPass.cpp's own (anonymous-namespace)
// ParticleRenderParamsUBO -- identical declaration to ParticleRender.vert's own copy (same set/
// binding, since this pipeline reuses that exact same pipeline layout).
layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
} g_Params;

layout(push_constant) uniform ParticleMeshRenderPC {
    uint expectedArchetype; // renderer::ParticleSystemPass::kMeshArchetypeBox/kMeshArchetypeIcosphere -- see this file's own header comment.
} pc;

// Real vertex input -- struct_custo.glsl's Vertex layout (position at byte 0, normal at byte 16,
// stride 48), matching geom_box.comp/geom_icosphere.comp's own generated output exactly (see this
// file's own header comment on why a real vertex buffer is used here, unlike the billboard/ribbon
// render modes).
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outColor;
layout(location = 3) out float outNormalizedAge;

void main() {
    uint particleIndex = sortedPairs[gl_InstanceIndex].index;
    Particle p = particles[particleIndex];
    uint kind = UnpackParticleKind(p.randomSeed);

    // Only kKindEmber particles ever carry a meaningful renderMode (precipitation's emitterIndex is
    // an unused default 0, see Particle::emitterIndex's own comment -- it must never be
    // reinterpreted as "this rain/snow particle wants to be a mesh instance just because EMITTERS[0]
    // happens to be in Mesh Particle mode"). A dead slot (p.life <= 0, recycled but not yet
    // overwritten by next frame's spawn) must also never draw.
    bool isMeshCandidate = (kind == kParticleKindEmber) && (p.life > 0.0);
    EmitterParams e = emitters[isMeshCandidate ? p.emitterIndex : 0u];
    bool matches = isMeshCandidate && (e.renderMode == 1u) && (e.meshArchetype == pc.expectedArchetype);

    if (!matches) {
        // Degenerate, always-clipped position (outside every NDC axis after the perspective
        // divide) -- see this file's own header comment for why this is a per-instance discard
        // rather than a separate compaction pass.
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outWorldPos = vec3(0.0);
        outNormal = vec3(0.0, 1.0, 0.0);
        outColor = vec4(0.0);
        outNormalizedAge = 0.0;
        return;
    }

    // Facing-velocity orthonormal basis -- B1's own "simple rotation" contract (face the direction
    // of travel) rather than a full physically-simulated tumble, matching this roadmap step's own
    // "keep it minimal" instruction. Falls back to a fixed forward axis when velocity is (near) zero
    // (a particle at/near spawn, or one whose forces have momentarily cancelled out) so the basis
    // never degenerates to NaN.
    vec3 forward = p.velocity;
    float speed = length(forward);
    forward = speed > 1.0e-4 ? (forward / speed) : vec3(0.0, 0.0, 1.0);
    // Guard against forward being (near-)parallel to the world-up hint, which would otherwise make
    // cross(worldUpHint, forward) degenerate (near-zero length) -- swap in a different hint axis
    // for that (rare, but real for a particle launched nearly straight up/down) case.
    vec3 worldUpHint = (abs(forward.y) > 0.999) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(worldUpHint, forward));
    vec3 up = cross(forward, right);

    // p.size.x is this particle's own CURRENT (colorCurve/sizeCurve-modulated every frame by
    // ParticleSimulation.comp's own UpdateParticle) uniform scale -- both generated meshes span
    // [-0.5, 0.5] per local axis, matching the exact same "size == full world-space extent"
    // convention ParticleRender.vert's own billboard offset already uses (corner - 0.5, scaled by
    // size), so this keeps "the emitter's own size curve modulates every render mode" true with no
    // extra per-archetype scale-remapping constant.
    float scale = p.size.x;
    vec3 localPos = inPosition * scale;
    vec3 worldPos = p.position + right * localPos.x + up * localPos.y + forward * localPos.z;

    // (right, up, forward) is an orthonormal basis, so rotating the normal by the same 3 columns
    // needs no inverse-transpose (a pure rotation matrix's inverse-transpose is itself).
    vec3 worldNormal = normalize(right * inNormal.x + up * inNormal.y + forward * inNormal.z);

    outWorldPos = worldPos;
    outNormal = worldNormal;
    // p.color already reflects EmitterParams.colorCurve sampled at this particle's own current
    // normalized age (ParticleSimulation.comp's own UpdateParticle, ember branch) -- exactly the
    // same per-frame curve-modulated value ParticleRender.vert's own outColor uses, so "the
    // emitter's own color curve still modulates it" holds for mesh particles with zero extra code
    // here.
    outColor = p.color;
    outNormalizedAge = clamp(p.life / max(p.maxLife, 1.0e-5), 0.0, 1.0);

    gl_Position = g_Params.viewProj * vec4(worldPos, 1.0);
}
