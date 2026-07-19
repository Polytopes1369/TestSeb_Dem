#version 460
#extension GL_GOOGLE_include_directive : enable

// Feature F7 (UE5.7/5.8 Niagara parity: shadow-casting particles): depth-only capture of every
// alive particle belonging to a castShadows-enabled emitter (renderer::ParticleSystemPass::
// EmitterParams::castShadows) into whichever renderer::VirtualShadowMapPass page this draw call
// targets -- see that class' own RecordParticleShadows() comment for the full per-page
// overlap-tested budget and explicit-barrier contract this vertex shader's inputs depend on.
//
// Unlike ParticleRender.vert (which reads Subtask 3's SORTED alive-particle list -- back-to-front
// order matters there for alpha blending), this reads the UNSORTED AliveListBuffer directly: sort
// order is irrelevant for a depth-only, discard-based-alpha-test capture (no blending; the
// hardware depth test does not care what order triangles arrive in), and using the raw list avoids
// taking any dependency on ParticleSystemPass's own separate sort-pass buffers/timing.
//
// gl_VertexIndex (0-5, two triangles, no bound vertex/index buffer) builds a QUAD exactly like
// ParticleRender.vert's own kQuadCorners table, but oriented to face the LIGHT (pc.light*, this
// page's own light-space basis -- see renderer::VirtualShadowMapPass's own class comment on why the
// CPU side reconstructs a TRUE, un-flipped up vector specifically for this billboard use, distinct
// from its page-grid-arithmetic-only lightUpForPaging) instead of the camera. gl_InstanceIndex
// indexes AliveListBuffer[0..aliveCount) -- EVERY alive particle is instanced; one whose OWN
// emitter does not have castShadows set is pushed to a degenerate always-clipped position instead
// of being excluded via a separate compacted list, the same "cheap per-vertex reject, no extra
// compaction pass" convention ParticleRender.vert's own render-mode gating (B1/B2/B3) already
// established for an analogous per-particle skip -- see that shader's own header comment.

#include "include/ParticleCommon.glsl"

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj;
    // Flat floats, not vec3 -- avoids vec3's implicit push-constant alignment padding surprises when
    // reasoning about the byte layout by eye, same convention renderer::ParticleSystemPass::
    // EmitterParams' own header comment establishes for its own struct.
    float lightRightX, lightRightY, lightRightZ;
    float lightUpX, lightUpY, lightUpZ;
} pc;

layout(location = 0) out vec2 outUV;

const vec2 kQuadCorners[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));

void main() {
    uint particleIndex = aliveIndices[gl_InstanceIndex];
    Particle p = particles[particleIndex];
    uint kind = UnpackParticleKind(p.randomSeed);

    // Only kKindEmber particles have a real EmitterParams slot of their own (see Particle::
    // emitterIndex's own declaration comment, ParticleCommon.glsl) -- precipitation (rain/snow)
    // never casts a VSM shadow in this feature's scope (matches the task's own "camera-facing
    // quads" billboard-only wording; a future step could add it via a dedicated flag on
    // PrecipitationParamsUBO instead, since precipitation has no EmitterParams slot to read one
    // from). Either way, degenerate the vertex if this particle should not cast a shadow.
    bool shouldCast = (kind == kParticleKindEmber) && (emitters[p.emitterIndex].castShadows != 0u);
    if (!shouldCast) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // Same always-clipped sentinel ParticleRender.vert's own render-mode gate uses.
        outUV = vec2(0.0);
        return;
    }

    vec2 corner = kQuadCorners[gl_VertexIndex];
    vec2 offset = corner - 0.5;

    vec3 lightRight = vec3(pc.lightRightX, pc.lightRightY, pc.lightRightZ);
    vec3 lightUp = vec3(pc.lightUpX, pc.lightUpY, pc.lightUpZ);

    // Light-facing billboard (not camera-facing like ParticleRender.vert, not velocity-aligned) --
    // the whole point of a shadow-caster footprint is to occlude the LIGHT's own view of the scene
    // behind it, so its plane must be perpendicular to the light direction, not whatever the
    // player's camera currently happens to be looking at.
    vec3 worldPos = p.position + offset.x * p.size.x * lightRight + offset.y * p.size.y * lightUp;

    outUV = corner;
    gl_Position = pc.lightViewProj * vec4(worldPos, 1.0);
}
