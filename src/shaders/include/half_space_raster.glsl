#ifndef HALF_SPACE_RASTER_GLSL
#define HALF_SPACE_RASTER_GLSL

// Shared half-space (edge function) triangle test, used by both ClusterSoftwareRaster.comp (to
// rasterize a micro-triangle cluster pixel-by-pixel) and ClusterResolve.comp (to reconstruct the
// barycentric coordinates of the exact pixel a rasterizer -- hardware or software -- already
// determined is covered by a given triangle). Factored into one shared copy so the "inside
// triangle" test and the barycentric-weight test always agree bit-for-bit between the pass that
// decided a pixel is covered and the pass that later re-derives its barycentric weights.
//
// E(a, b, p) is twice the signed area of the triangle (a, b, p): positive if p is to the left of
// the directed edge a->b (CCW winding in screen space, which is Y-down in Vulkan), negative if to
// the right, zero if p is exactly on the line through a and b. For a triangle (v0, v1, v2), the
// three per-edge evaluations E(v1, v2, p), E(v2, v0, p), E(v0, v1, p) are, up to a shared constant
// factor of 1 / E(v0, v1, v2), exactly the barycentric weights (w0, w1, w2) of p with respect to
// (v0, v1, v2) -- and p lies inside the triangle iff all three share the same sign as
// E(v0, v1, v2) itself (or any one is exactly zero, on an edge). This is the standard half-space
// rasterization test (Pineda 1988), computed here directly from screen-space (pixel) coordinates
// rather than any fixed-point/subpixel-precision representation -- adequate for micro-triangles
// covering only a handful of pixels, where subpixel precision error is negligible relative to a
// single pixel's footprint.
float EdgeFunction(vec2 a, vec2 b, vec2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// Standard top-left fill-rule tie-break (the rule real GPU rasterizer hardware applies
// automatically, and that renderer::ClusterSoftwareRasterPass's manual half-space test must
// replicate by hand): a pixel exactly on a shared edge between two adjacent triangles must be
// claimed by exactly ONE of them, never both and never neither, or the two triangles' coverage
// tests can disagree right at their shared boundary. A purely inclusive (>=0) test on both sides of
// a shared edge fails this -- both triangles independently see w==0 as "inside" -- which is exactly
// the mechanism behind the shimmering/z-fighting seams this function fixes (see
// cluster_software_raster_core.glsl's own call site).
//
// "Top" edge: exactly horizontal, pointing in +X (screen space is Y-down in Vulkan, so this is the
// edge running along the top of the triangle in the conventional sense). "Left" edge: pointing in
// -Y (i.e. upward on screen). `edgeStart`/`edgeEnd` must be passed in the SAME winding order the
// edge function itself was evaluated with (e.g. EdgeFunction(a, b, p) pairs with
// IsTopLeftEdge(a, b), not (b, a)) -- reversing the pair negates the edge vector, which negates
// this function's own result, which is exactly what makes the rule antisymmetric: the two triangles
// sharing a physical edge always evaluate it in opposite vertex order, so of the two, exactly one
// ever satisfies this tie-break.
bool IsTopLeftEdge(vec2 edgeStart, vec2 edgeEnd) {
    vec2 e = edgeEnd - edgeStart;
    return (e.y == 0.0 && e.x > 0.0) || (e.y < 0.0);
}

#endif // HALF_SPACE_RASTER_GLSL
