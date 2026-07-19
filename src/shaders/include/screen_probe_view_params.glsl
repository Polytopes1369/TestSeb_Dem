#ifndef SCREEN_PROBE_VIEW_PARAMS_GLSL
#define SCREEN_PROBE_VIEW_PARAMS_GLSL

// Byte-for-byte std140 mirror of renderer::ScreenProbeViewParamsUBO (ScreenProbeGIPass.cpp),
// shared by every stage of the screen-probe ping-pong pipeline (ScreenProbeTrace.comp,
// ScreenProbeTemporal.comp, ScreenProbeGather.comp) so the layout can never silently drift
// between independently hand-typed copies. Each stage uses a different subset of fields (see
// each shader's own "Unused here" comments); the struct itself is identical everywhere.
//
// The includer must #define SCREEN_PROBE_VIEW_PARAMS_SET / SCREEN_PROBE_VIEW_PARAMS_BINDING
// before including this header.
layout(std140, set = SCREEN_PROBE_VIEW_PARAMS_SET, binding = SCREEN_PROBE_VIEW_PARAMS_BINDING) uniform ScreenProbeViewParamsUBO {
    mat4 invViewProj;
    mat4 prevViewProj;
    vec2 viewportSize;
    vec2 probeGridSize;
} g_ViewParams;

#endif // SCREEN_PROBE_VIEW_PARAMS_GLSL
