#version 460
#extension GL_EXT_ray_tracing : require

// Part 2 deliverable: Miss Shader. A ray that hits nothing in the Fallback Mesh TLAS carries no
// Surface Cache color -- the caller (SurfaceCacheHWRT.rgen, or a future final-gather consumer)
// distinguishes this from "hit but the surface's radiance happens to be black" via the explicit
// hit flag, matching SurfaceCacheTraceSWRT.comp's identical miss convention exactly.

struct RayResult {
    vec3 color;
    float hit;
};
layout(location = 0) rayPayloadInEXT RayResult g_Payload;

void main() {
    g_Payload.color = vec3(0.0);
    g_Payload.hit = 0.0;
}
