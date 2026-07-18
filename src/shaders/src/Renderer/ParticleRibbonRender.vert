#version 460
#extension GL_GOOGLE_include_directive : enable

// Niagara-parity roadmap, subtask B2 (Ribbon/Trail render mode): draws every alive kKindEmber
// particle whose OWN emitter has EmitterParams.renderMode == 2 (Ribbon/Trail) as a camera-facing
// quad-strip built from that particle's own recent world-position history (renderer::
// ParticleSystemPass's per-slot ring buffer, see ParticleRibbonCommon.glsl's own header comment) --
// no bound vertex buffer, exactly like ParticleRender.vert's own billboard quad: every corner of
// every segment is generated from gl_VertexIndex alone via the SAME kQuadCorners lookup table idiom.
//
// This pipeline reuses the billboard pipeline's own 4 sets (0 = particle state, 1 =
// SortedPairsBuffer, 2 = ParticleRenderParamsUBO + scene depth sampler, 3 = VSM + World Probe Grid
// lighting -- ribbons are alpha-blended and back-to-front sorted exactly like billboards, unlike
// ParticleMeshRender's opaque solid instances) plus a 5th, NEW set (4 = ribbon position history).
//
// Geometry: kRibbonHistorySamples (6) position samples per particle make (kRibbonHistorySamples - 1)
// == 5 segments, each a 6-vertex (2-triangle) quad -- 30 fixed vertices per instance
// (renderer::ParticleSystemPass::RecordDraw's own m_RibbonIndirectDrawBuffer.vertexCount). Sample 0
// (ago == 0) is always this particle's OWN CURRENT position (already pushed into ring slot
// (ribbonSampleCount - 1) by ParticleSimulation.comp's own UpdateParticle THIS SAME FRAME, before
// this draw runs), so the ribbon's head always exactly tracks the particle with zero extra lag.
// Samples further from ring index 0 (ago 1..5) are progressively older -- a particle that has not
// lived long enough yet to have accumulated the full 6 samples simply collapses its trailing
// (unused, "too old") segments to zero length (see GetRibbonSample's own comment) instead of reading
// another particle's stale ring contents.
//
// Width: each segment's cross-section direction is derived from ITS OWN tangent (direction between
// its 2 endpoint samples) crossed with the direction to the camera -- the standard "screen-facing
// ribbon" technique (same visual family as a classic motion-trail/comet-tail effect), so the strip
// always presents its widest face toward the viewer rather than being drawn edge-on and vanishing.

#include "include/ParticleCommon.glsl"
#define RIBBON_HISTORY_SET 4
#include "include/ParticleRibbonCommon.glsl"

struct SortedPair {
    uint index;
    float key;
};
layout(std430, set = 1, binding = 0) readonly buffer SortedPairsBuffer {
    SortedPair sortedPairs[];
};

layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
} g_Params;

const vec2 kQuadCorners[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec4 outColor;
layout(location = 2) out float outNormalizedAge;   // Particle's own remaining-life fraction (life fade, same as billboard).
layout(location = 3) out float outTailFade;         // [0,1] -- 1.0 at the ribbon's head (this frame), fading to 0.0 at its oldest valid sample.
layout(location = 4) out float outWidthCoord;       // [0,1] across the ribbon's own cross-section -- ParticleRibbonRender.frag soft-fades the 2 edges from this, same "no texture assets, analytic shape mask" convention as ParticleRender.frag's own soft-circle.

// Reads this particle's own ring buffer at `ago` samples before its current (freshest) one, clamped
// to however many samples are ACTUALLY valid so far (a young particle's unused, too-far-back
// history simply repeats its own oldest real sample -- collapsing that segment to zero length rather
// than reading stale data from this slot's previous occupant, see ParticleRibbonCommon.glsl's own
// RibbonSampleCountBuffer comment).
vec3 GetRibbonSample(uint particleIndex, uint ago, uint validCount) {
    uint clampedAgo = min(ago, validCount - 1u); // validCount >= 1 always for a live particle (spawn seeds it to 1, see SpawnParticle's own comment).
    uint totalPushed = ribbonSampleCount[particleIndex];
    uint ringIndex = (totalPushed - 1u + kRibbonHistorySamples - clampedAgo) % kRibbonHistorySamples;
    return ribbonHistory[particleIndex * kRibbonHistorySamples + ringIndex].xyz;
}

void main() {
    uint particleIndex = sortedPairs[gl_InstanceIndex].index;
    Particle p = particles[particleIndex];
    uint kind = UnpackParticleKind(p.randomSeed);

    // Same render-mode gating contract as ParticleMeshRender.vert's own (see that shader's own
    // comment) -- only kKindEmber particles whose own emitter is in Ribbon mode draw here; every
    // other instance is pushed to a degenerate, always-clipped position.
    bool isRibbonCandidate = (kind == kParticleKindEmber) && (p.life > 0.0);
    EmitterParams e = emitters[isRibbonCandidate ? p.emitterIndex : 0u];
    bool matches = isRibbonCandidate && (e.renderMode == 2u);

    if (!matches) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outWorldPos = vec3(0.0);
        outColor = vec4(0.0);
        outNormalizedAge = 0.0;
        outTailFade = 0.0;
        outWidthCoord = 0.0;
        return;
    }

    uint validCount = min(ribbonSampleCount[particleIndex], kRibbonHistorySamples);

    uint segmentIndex = gl_VertexIndex / 6u; // 0 .. kRibbonHistorySamples - 2.
    uint cornerIndex = gl_VertexIndex % 6u;
    vec2 corner = kQuadCorners[cornerIndex];

    uint agoNear = segmentIndex;
    uint agoFar = segmentIndex + 1u;
    vec3 nearPos = GetRibbonSample(particleIndex, agoNear, validCount);
    vec3 farPos = GetRibbonSample(particleIndex, agoFar, validCount);
    uint ago = (corner.x < 0.5) ? agoNear : agoFar;
    vec3 samplePos = (corner.x < 0.5) ? nearPos : farPos;

    // Screen-facing width direction -- this segment's own tangent (direction between its 2
    // endpoints) crossed with the direction to the camera. Falls back to the particle's own velocity
    // direction (and, failing that, the camera's own right vector) when the segment has collapsed to
    // zero length (a not-yet-valid trailing segment, see GetRibbonSample's own comment) or happens to
    // point directly at the camera, both of which would otherwise make the cross product degenerate.
    vec3 segmentVec = nearPos - farPos;
    float segmentLen = length(segmentVec);
    vec3 tangent = segmentLen > 1.0e-5 ? (segmentVec / segmentLen) : normalize(p.velocity + vec3(0.0, 0.0, 1.0e-4));
    vec3 toCamera = g_Params.cameraPosition - samplePos;
    float toCameraLen = length(toCamera);
    vec3 toCameraDir = toCameraLen > 1.0e-5 ? (toCamera / toCameraLen) : g_Params.cameraRight;
    vec3 widthDir = cross(tangent, toCameraDir);
    float widthDirLen = length(widthDir);
    widthDir = widthDirLen > 1.0e-5 ? (widthDir / widthDirLen) : g_Params.cameraRight;

    // e.ribbonWidth is already documented as a HALF-width (see EmitterParams::ribbonWidth's own
    // declaration comment) -- corner.y in {0, 1} maps to {-1, +1} directly, no extra 0.5 factor.
    float side = corner.y * 2.0 - 1.0;
    vec3 worldPos = samplePos + widthDir * (e.ribbonWidth * side);

    outWorldPos = worldPos;
    // p.color already reflects this particle's OWN current colorCurve sample (ParticleSimulation.
    // comp's own UpdateParticle) -- the SAME per-frame curve-modulated value every render mode uses,
    // applied uniformly across the whole ribbon (a full per-sample color HISTORY would need its own
    // additional history buffer, out of scope for this roadmap step -- see this shader's own header
    // comment) while `outTailFade` below still gives the trail a visually distinct head-to-tail fade.
    outColor = p.color;
    outNormalizedAge = clamp(p.life / max(p.maxLife, 1.0e-5), 0.0, 1.0);
    outTailFade = 1.0 - clamp(float(ago) / float(max(validCount - 1u, 1u)), 0.0, 1.0);
    outWidthCoord = corner.y;

    gl_Position = g_Params.viewProj * vec4(worldPos, 1.0);
}
