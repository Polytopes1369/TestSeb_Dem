#version 460

// Debug-only text overlay fragment shader (see DebugText.vert's own comment on why this file
// compiles unconditionally but only ever runs from Debug-only C++ code paths). Reads one bit per
// texel from the font bitmask SSBO (renderer::debug::BitmapFont8x8.h's table, uploaded once at
// renderer::debug::DebugTextOverlay::Init) -- no sampler, no image, just an indexed bit test, so
// no image asset and no additional descriptor type is needed for this whole feature.

layout(std430, set = 0, binding = 1) readonly buffer FontBitmapSSBO {
    // 128 characters x 8 rows, flattened: row R of character C lives at index (C * 8 + R). Only
    // the low 8 bits of each entry are meaningful (bit 7 = the glyph's leftmost pixel).
    uint rows[];
} g_Font;

layout(location = 0) in vec2 inGlyphLocalPixel;
layout(location = 1) flat in uint inCharCode;

layout(location = 0) out vec4 outColor;

void main() {
    ivec2 texel = ivec2(floor(inGlyphLocalPixel));
    if (texel.x < 0 || texel.x >= 8 || texel.y < 0 || texel.y >= 8) {
        discard;
    }

    uint rowBits = g_Font.rows[inCharCode * 8u + uint(texel.y)];
    uint bit = (rowBits >> (7u - uint(texel.x))) & 1u;
    if (bit == 0u) {
        discard;
    }

    // Fixed bright yellow, matching this codebase's existing debug-text convention (see
    // ClusterResolve.comp's own DEBUG view-mode string drawing, DrawString/finalColor = (1,1,0)).
    outColor = vec4(1.0, 1.0, 0.0, 0.9);
}
