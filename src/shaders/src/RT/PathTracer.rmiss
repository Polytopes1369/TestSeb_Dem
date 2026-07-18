#version 460
#extension GL_EXT_ray_tracing : require

// UE5.8 rendering-parity gap G10b -- reference Path Tracer miss shader (DEBUG-only, see
// PathTracer.rgen's header). A scene ray that hits nothing just flags a miss (hitT < 0.0); the
// ray-gen shader computes the analytic sky-dome radiance itself from the ray direction (see
// PathTracer.rgen's SampleSky), so no sky evaluation is needed here.

struct PTPayload {
    vec3 hitPos;
    vec3 hitNormal;
    uint materialID;
    float hitT;
};
layout(location = 0) rayPayloadInEXT PTPayload g_Payload;

void main() {
    g_Payload.hitT = -1.0;
}
