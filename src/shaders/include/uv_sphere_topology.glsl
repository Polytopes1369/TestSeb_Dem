#ifndef UV_SPHERE_TOPOLOGY_GLSL
#define UV_SPHERE_TOPOLOGY_GLSL

// Shared UV-sphere grid index topology (top cap fan, middle quads, bottom cap fan) for the
// "one compute invocation per vertex, each invocation also conditionally writes 0-3 triangle
// index blocks based on its own ID range" pattern used by every PrimitiveGen shader built on a
// (lat, lon) sample grid -- geom_sphere.comp and geom_chamferBox.comp reshape the same grid's
// vertex positions differently (plain unit sphere vs. superellipsoid box), but the triangle
// connectivity between grid vertices is identical, so only this function is shared, not vertex
// generation. The includer must already have an `indices` writeonly uint[] SSBO in scope
// (typically `layout(std430, binding = 1) writeonly buffer IndexBuffer { uint indices[]; };`,
// declared before this header is included).
void WriteUVSphereTopologyIndices(uint indexOffset, uint vertexOffset, uint id, uint sideSegs, uint ringCount, uint vertCount) {
    if (id < sideSegs) // top cap fan
    {
        uint lon = id;
        uint tIdx = indexOffset + lon * 3u;
        indices[tIdx + 0u] = vertexOffset + lon + 2u;
        indices[tIdx + 1u] = vertexOffset + lon + 1u;
        indices[tIdx + 2u] = vertexOffset;
    }

    if (ringCount > 1u && id < (ringCount - 1u) * sideSegs) // middle quads
    {
        uint lat = id / sideSegs;
        uint lon = id % sideSegs;
        uint current = lon + lat * (sideSegs + 1u) + 1u;
        uint next    = current + (sideSegs + 1u);

        uint tIdx = indexOffset + sideSegs * 3u + (lat * sideSegs + lon) * 6u;
        indices[tIdx + 0u] = vertexOffset + current;
        indices[tIdx + 1u] = vertexOffset + current + 1u;
        indices[tIdx + 2u] = vertexOffset + next + 1u;
        indices[tIdx + 3u] = vertexOffset + current;
        indices[tIdx + 4u] = vertexOffset + next + 1u;
        indices[tIdx + 5u] = vertexOffset + next;
    }

    if (id < sideSegs) // bottom cap fan
    {
        uint lon = id;
        uint tIdx = indexOffset + sideSegs * 3u + (ringCount - 1u) * sideSegs * 6u + lon * 3u;
        indices[tIdx + 0u] = vertexOffset + vertCount - 1u;
        indices[tIdx + 1u] = vertexOffset + vertCount - (lon + 2u) - 1u;
        indices[tIdx + 2u] = vertexOffset + vertCount - (lon + 1u) - 1u;
    }
}

#endif // UV_SPHERE_TOPOLOGY_GLSL
