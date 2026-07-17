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

bool hasBit(int x, int y) {
    int fontX = int(floor(float(x) / 2.0));
    int fontY = int(floor(float(y) / 2.0));
    if (fontX < 0 || fontX >= 8 || fontY < 0 || fontY >= 8) return false;
    uint rowBits = g_Font.rows[inCharCode * 8u + uint(fontY)];
    return ((rowBits >> (7u - uint(fontX))) & 1u) != 0u;
}

void main() {
    ivec2 texel = ivec2(floor(inGlyphLocalPixel));

    // 1. Check if the center pixel itself is set (White text)
    if (hasBit(texel.x, texel.y)) {
        outColor = vec4(1.0, 1.0, 1.0, 0.9); // White text with 0.9 alpha
        return;
    }

    // 2. Check if any neighboring pixel within 2px is set (Black outline)
    bool isOutline = false;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            if (dx == 0 && dy == 0) continue;
            // 2px circle radius: dx^2 + dy^2 <= 5
            if (dx * dx + dy * dy <= 5) {
                if (hasBit(texel.x + dx, texel.y + dy)) {
                    isOutline = true;
                    break;
                }
            }
        }
        if (isOutline) break;
    }

    if (isOutline) {
        outColor = vec4(0.0, 0.0, 0.0, 0.9); // Black outline with 0.9 alpha
    } else {
        discard;
    }
}
