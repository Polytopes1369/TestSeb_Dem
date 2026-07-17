#version 460

// Debug-only text overlay (renderer::debug::DebugTextOverlay) -- draws one axis-aligned 8x8-pixel
// quad per glyph instance, entirely from gl_VertexIndex/gl_InstanceIndex (no vertex buffer, same
// "bindless" convention as every other pipeline in this codebase). This whole pipeline is only
// ever created/dispatched from Debug-only C++ code (#ifndef NDEBUG), so it never ships in the
// Release .exe (CLAUDE.md's debug-tooling exclusion rule) even though this .comp/.vert/.frag file
// itself is compiled into shaders/ unconditionally by CMake's blanket shader-compile step.

struct GlyphInstance {
    vec2 screenPosPixels; // Top-left corner of this glyph's 8x8 cell, in framebuffer pixels.
    uint charCode;         // ASCII code, indexes FontBitmapSSBO (DebugText.frag) 1:1.
    uint _pad;
};

layout(std430, set = 0, binding = 0) readonly buffer GlyphInstancesSSBO {
    GlyphInstance instances[];
} g_Glyphs;

layout(push_constant) uniform DebugTextPushConstants {
    vec2 viewportSize;
} pc;

layout(location = 0) out vec2 outGlyphLocalPixel; // 0..8 local pixel coords within this glyph's cell.
layout(location = 1) flat out uint outCharCode;

// 2 triangles (6 vertices), no index buffer -- corner order matches standard CCW winding for a
// quad spanning [0,1]^2 in the glyph's own local space.
const vec2 kQuadCorners[6] = vec2[6](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

const float kGlyphPixelSize = 16.0; // 8x8 font (BitmapFont8x8.h), drawn at 2:1 pixel scale.

void main() {
    GlyphInstance inst = g_Glyphs.instances[gl_InstanceIndex];
    vec2 corner = kQuadCorners[gl_VertexIndex];

    // Expand local quad size by 2 pixels on each border to make room for a 2px black outline
    float kPadding = 2.0;
    vec2 localPos = -kPadding + corner * (kGlyphPixelSize + 2.0 * kPadding);
    vec2 pixelPos = inst.screenPosPixels + localPos;

    // Pixel space (origin top-left, +Y down) -> NDC (origin center, +Y down in Vulkan) needs only
    // a scale + bias, no Y flip -- Vulkan's own y-down clip space already matches framebuffer
    // pixel-space orientation.
    vec2 ndc = (pixelPos / pc.viewportSize) * 2.0 - 1.0;

    outGlyphLocalPixel = localPos;
    outCharCode = inst.charCode;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
