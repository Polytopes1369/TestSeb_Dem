#ifndef FALLBACK_GEOMETRY_GLSL
#define FALLBACK_GEOMETRY_GLSL

// Combined Fallback Mesh vertex/index/draw-range buffers -- the SAME buffers every BLAS in the
// scene's TLAS was built directly against (renderer::SurfaceCachePass::GetVertexBuffer()/
// GetIndexBuffer(), renderer::SurfaceCacheRayTracingPass::GetDrawRangeBuffer(); see
// renderer::AccelerationStructure::BuildBLAS's own comment on why no separate geometry upload
// exists for ray tracing). Promoted from what used to be 5 independent copy-pasted struct+binding
// blocks (SurfaceCacheHWRT.rchit, ReflectionTrace.comp, ScreenProbeTrace.comp,
// SurfaceCacheGIInject.comp, WorldProbeInject.comp) into one shared include -- a field-order
// mismatch between two copies would have silently corrupted every HWRT/SWRT fallback trace.
//
// The includer must #define FALLBACK_GEOMETRY_SET / FALLBACK_GEOMETRY_BASE_BINDING before
// including this header. Declares 3 consecutive bindings at the given set, starting at
// FALLBACK_GEOMETRY_BASE_BINDING: FallbackVertexBuffer (+0), FallbackIndexBuffer (+1),
// EntityDrawRangeBuffer (+2) -- matching renderer::VulkanUtils::WriteSharedGeometryBindings'
// C++-side TLAS+vertex+index+drawRange write convention (TLAS itself is bound separately by each
// includer, since not every consumer of this header uses the same TLAS binding slot).

// std430-exact mirror of geometry::FallbackVertex (32 bytes: 3+3+2 floats, no padding -- see
// ClusterFormat.h's own static_assert) using flat scalars instead of vec3 fields, to avoid
// std430's vec3-alignment padding diverging from the tightly-packed C++ layout.
struct FallbackVertexGpu {
    float posX, posY, posZ;
    float normX, normY, normZ;
    float u, v;
};
layout(std430, set = FALLBACK_GEOMETRY_SET, binding = FALLBACK_GEOMETRY_BASE_BINDING) readonly buffer FallbackVertexBuffer {
    FallbackVertexGpu g_Vertices[];
};
layout(std430, set = FALLBACK_GEOMETRY_SET, binding = FALLBACK_GEOMETRY_BASE_BINDING + 1) readonly buffer FallbackIndexBuffer {
    uint g_Indices[];
};

// std430-exact mirror of renderer::SurfaceCachePass::EntityDrawRange (vertexOffset/firstIndex/
// indexCount, 12 bytes in C++ but padded to 16 here for std430's own array-stride rounding --
// the C++-side upload zero-fills the pad field, see SurfaceCacheRayTracingPass::Init()).
struct EntityDrawRangeGpu {
    int vertexOffset;
    uint firstIndex;
    uint indexCount;
    uint _pad;
};
layout(std430, set = FALLBACK_GEOMETRY_SET, binding = FALLBACK_GEOMETRY_BASE_BINDING + 2) readonly buffer EntityDrawRangeBuffer {
    EntityDrawRangeGpu g_DrawRanges[];
};

#endif // FALLBACK_GEOMETRY_GLSL
