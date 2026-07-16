#version 460
#extension GL_GOOGLE_include_directive : enable

// Surface Cache capture fragment shader (see renderer::SurfaceCachePass): writes albedo/normal/
// emissive for one texel of the global surface-cache atlas, at whatever position within the
// currently-bound Card's exclusive rect this invocation's pixel maps to (the render area/viewport/
// scissor are all set to exactly that rect by RecordCapture(), so gl_FragCoord never needs to be
// consulted here). This codebase has no texture/material-binding system (see ClusterResolve.comp's
// own comment) -- this reuses the exact same procedural-material approach every other shading pass
// already uses (procedural_material.glsl's HashID/HsvToRgb, keyed by entityID) plus a small
// triplanar value-noise modulation so a captured card is not perfectly flat-shaded.

#include "include/procedural_material.glsl"
#include "include/math_utils.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;

layout(push_constant) uniform SurfaceCaptureConstants {
    mat4 viewProj;
    uint entityID;
} pc;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outEmissive;
// Seeded outgoing-radiance atlas texel (renderer::SurfaceCachePass::kRadianceFormat) -- a small
// constant-ambient proxy on the albedo plus this card's emissive tint, so a SWRT/HWRT trace
// sampling this texel before any SurfaceCacheGIInject.comp pass has run over it still reads a
// plausible non-black value instead of true zero. kInitialRadianceAmbientProxy mirrors the C++
// constant of the same name (renderer::SurfaceCachePass.h) exactly.
layout(location = 3) out vec4 outRadiance;
// World-space (== local-space here -- see this shader's own header comment) hit position, full
// float precision: the 3D origin a GI injection pass fires its hemisphere rays from.
layout(location = 4) out vec4 outWorldPos;

const float kInitialRadianceAmbientProxy = 0.15;

// Octahedral encoding of a unit vector into [0,1]^2 -- the same compact normal encoding this
// codebase already uses for cluster vertex normals (geometry::ClusterVertexNormal /
// GeometryEncoding.h), applied here to the captured world-space surface normal.
vec2 OctEncode(vec3 n) {
    vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z < 0.0) {
        vec2 signP = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * signP;
    }
    return p * 0.5 + 0.5;
}

void main() {
    vec3 n = normalize(inWorldNormal);

    // Stable per-entity hue (matches ClusterResolve.comp / draw.frag's own procedural look), plus
    // a small triplanar value-noise modulation from world position so a card is not perfectly
    // flat-colored.
    float hue = float(HashID(pc.entityID) & 0xFFFFu) / 65536.0;
    vec3 baseColor = HsvToRgb(vec3(hue, 0.55, 0.85));

    vec3 blend = abs(n);
    blend /= (blend.x + blend.y + blend.z + 1e-5);
    float noise =
        Hash(inWorldPos.yz) * blend.x +
        Hash(inWorldPos.xz) * blend.y +
        Hash(inWorldPos.xy) * blend.z;
    vec3 albedo = baseColor * mix(0.85, 1.15, noise);

    outAlbedo = vec4(clamp(albedo, 0.0, 1.0), 1.0);
    outNormal = vec4(OctEncode(n), 0.0, 1.0);

    // Subtle procedural emissive tint, distinct per entity, so the channel is exercised
    // end-to-end rather than always writing zero -- no material system flags entities as
    // emissive/non-emissive yet (geometry::EntityMaterialProperties only carries WPO/mask data),
    // so every card gets the same small glow rather than an arbitrary on/off split.
    vec3 emissive = baseColor * 0.04;
    outEmissive = vec4(emissive, 1.0);

    outRadiance = vec4(emissive + albedo * kInitialRadianceAmbientProxy, 1.0);
    outWorldPos = vec4(inWorldPos, 1.0);
}
