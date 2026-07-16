#ifndef CLUSTER_CULLING_TESTS_GLSL
#define CLUSTER_CULLING_TESTS_GLSL

// Shared per-cluster frustum + backface (normal-cone) rejection tests. Factored out of
// ClusterFrustumCull.comp so ClusterHZBOcclusionCull.comp's early-pass specialization (which must
// perform the exact same two tests before ever consulting the HZB) can reuse this single,
// already-verified copy instead of a second, independently-maintained one that could silently
// drift out of sync. Every function takes its inputs explicitly (no dependency on any
// particular global UBO/SSBO name or binding), so it works unmodified regardless of the
// includer's own descriptor layout.

// True if the axis-aligned box [boundsMin, boundsMax] lies entirely on the negative (outside)
// side of `plane` (xyz = outward unit normal, w = signed distance) -- the standard "positive
// vertex" (p-vertex) test: of the box's 8 corners, the one furthest along the plane's normal is
// picked per-axis from whichever bound (min or max) is on the positive side of that axis; if even
// that most-favorable corner is behind the plane, every other corner is too.
bool BoxOutsidePlane(vec3 boundsMin, vec3 boundsMax, vec4 plane) {
    vec3 positiveVertex = vec3(
        plane.x >= 0.0 ? boundsMax.x : boundsMin.x,
        plane.y >= 0.0 ? boundsMax.y : boundsMin.y,
        plane.z >= 0.0 ? boundsMax.z : boundsMin.z);
    return dot(plane.xyz, positiveVertex) + plane.w < 0.0;
}

// AABB/frustum intersection test: the box is culled only if it is fully outside at least one of
// the 6 planes. A box straddling every plane (partially inside, or fully enclosing the frustum) is
// conservatively kept -- this is a coarse accept/reject test, not an exact clip.
bool IsBoxInsideFrustum(vec3 boundsMin, vec3 boundsMax, vec4 frustumPlanes[6]) {
    for (int i = 0; i < 6; ++i) {
        if (BoxOutsidePlane(boundsMin, boundsMax, frustumPlanes[i])) {
            return false;
        }
    }
    return true;
}

// Normal-cone backface test. coneAxis/coneCutoff bound every triangle normal in the cluster to
// dot(n, coneAxis) >= coneCutoff (see ClusterFormat.h / VirtualGeometryCacheTest.cpp's
// BuildIndexEntry, which is exactly how coneAxis/coneCutoff are computed on the CPU: coneAxis is
// the average outward-facing normal, coneCutoff the worst-case (minimum) dot product against it
// across every normal in the cluster).
//
// Let d = sphereCenter - cameraPosition (points from the camera toward the cluster) and
// dist = |d|. A triangle at surface point S with normal n is front-facing toward the camera iff
// dot(n, cameraPosition - S) > 0, i.e. iff n points at least partly back toward the camera.
// Bounding every possible S to the cluster's bounding sphere (|S - sphereCenter| <= sphereRadius)
// and every possible n to the cone (dot(n, coneAxis) >= coneCutoff), the cluster can be guaranteed
// to have no front-facing triangle only when even the most favorable (S, n) pair inside those
// bounds still fails the front-facing test, giving the conservative rejection condition:
//
//     dot(d, coneAxis) >= coneCutoff * dist + sphereRadius
//
// (the meshopt_Bounds/Niagara-style cluster cone test). Larger sphereRadius requires a stronger
// axis/view alignment before culling is allowed, correctly making larger clusters harder to reject
// outright than small tightly-bounded ones.
bool IsClusterBackFacing(vec3 sphereCenter, float sphereRadius, vec3 coneAxis, float coneCutoff, vec3 cameraPositionWorld) {
    vec3 cameraToCenter = sphereCenter - cameraPositionWorld;
    float dist = length(cameraToCenter);

    // Degenerate case: the camera sits at (or within floating-point epsilon of) the cluster's
    // bounding-sphere center -- no meaningful viewing direction exists, so conservatively treat
    // the cluster as front-facing (never cull) rather than divide by a near-zero distance.
    if (dist < 1.0e-5) {
        return false;
    }

    return dot(cameraToCenter, coneAxis) >= coneCutoff * dist + sphereRadius;
}

#endif // CLUSTER_CULLING_TESTS_GLSL
