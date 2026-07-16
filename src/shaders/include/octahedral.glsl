#ifndef OCTAHEDRAL_GLSL
#define OCTAHEDRAL_GLSL

// Standard octahedral encoding of a unit vector into [0,1]^2 -- promoted from what used to be 3
// independent copy-pasted implementations (SurfaceCacheCapture.frag's outNormal,
// SurfaceCacheGIInject.comp's OctDecode, and this feature's own new ClusterResolve.comp /
// ScreenProbeTrace.comp / ScreenProbeTemporal.comp consumers) into one shared include.

vec2 OctEncode(vec3 n) {
    vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z < 0.0) {
        vec2 signP = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * signP;
    }
    return p * 0.5 + 0.5;
}

vec3 OctDecode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.xy += mix(vec2(t, t), vec2(-t, -t), greaterThanEqual(n.xy, vec2(0.0)));
    return normalize(n);
}

#endif
