#ifndef REFLECTION_VIEW_PARAMS_GLSL
#define REFLECTION_VIEW_PARAMS_GLSL

// Byte-for-byte std140 mirror of renderer::ReflectionViewParamsUBO (ReflectionPass.cpp), shared
// by every stage of the reflection ping-pong pipeline (ReflectionTrace.comp, ReflectionTemporal
// .comp, ReflectionGather.comp) so the layout can never silently drift between independently
// hand-typed copies. Each stage uses a different subset of fields (see each shader's own "Unused
// here" comments); the struct itself is identical everywhere.
//
// The includer must #define REFLECTION_VIEW_PARAMS_SET / REFLECTION_VIEW_PARAMS_BINDING before
// including this header.
layout(std140, set = REFLECTION_VIEW_PARAMS_SET, binding = REFLECTION_VIEW_PARAMS_BINDING) uniform ReflectionViewParamsUBO {
    mat4 invViewProj;
    mat4 prevViewProj;
    vec3 cameraPositionWorld;
    float _pad0;
    vec2 viewportSize;
    float _pad1;
    float _pad2;
} g_ViewParams;

#endif // REFLECTION_VIEW_PARAMS_GLSL
