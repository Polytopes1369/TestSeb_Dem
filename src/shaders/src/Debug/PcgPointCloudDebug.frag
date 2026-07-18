#version 460

// Debug-only wireframe box gizmo per PCG point (renderer::debug::PcgPointCloudDebugView, PCG
// roadmap Phase 7.2) -- trivial unlit passthrough of PcgPointCloudDebug.vert's own density-derived
// color, opaque (alpha = 1.0, no blending: this pipeline's colorBlendAttachment.blendEnable is
// VK_FALSE, see PcgPointCloudDebugView.cpp's own Init()). Only ever created/dispatched from
// Debug-only C++ code (#ifndef NDEBUG); this file itself still compiles into shaders/
// unconditionally by CMake's blanket shader-compile step, same as every other Debug/*.frag file.

layout(location = 0) in vec3 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(inColor, 1.0);
}
