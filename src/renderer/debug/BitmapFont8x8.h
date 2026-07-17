#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): a minimal
// 8x8-per-glyph bitmap font covering exactly the character set renderer::DebugTextOverlay's own
// stat lines use (digits, uppercase letters, space/colon/slash/period/percent/dash) -- not a full
// 128-entry ASCII table, since nothing in this codebase needs one. Each glyph is built from a
// classic 5-column x 7-row dot-matrix pattern (the same shape used by countless character
// generator ROMs), left-shifted 3 bits so it occupies the high 5 bits of each row byte -- bit 7 is
// the glyph's leftmost pixel, matching DebugText.frag's own bit-test convention.
#ifndef NDEBUG

#include <array>
#include <cstdint>

namespace renderer::debug {

    // 128 entries (indexed directly by ASCII code) x 8 rows -- any character not explicitly listed
    // in BuildFont8x8() below stays all-zero (renders as blank), which is the desired behavior for
    // characters this overlay never actually emits.
    using Font8x8Table = std::array<std::array<uint8_t, 8>, 128>;

    namespace detail {

        // Packs a 5-bit-per-row glyph (bit4 = leftmost pixel) into this font's 8-bit-per-row
        // storage (bit7 = leftmost pixel, bits 2..0 unused) -- see the file comment.
        inline std::array<uint8_t, 8> MakeGlyph(std::initializer_list<uint8_t> rows5) {
            std::array<uint8_t, 8> glyph{};
            uint32_t row = 0;
            for (uint8_t bits5 : rows5) {
                if (row >= 8) break;
                glyph[row++] = static_cast<uint8_t>((bits5 & 0x1Fu) << 3);
            }
            return glyph;
        }

    } // namespace detail

    // Builds the sparse font table once (called only from renderer::debug::DebugTextOverlay::Init,
    // itself Debug-only) -- not a compile-time constexpr table, since a mostly-zero 128x8 array is
    // more clearly expressed as "set these ~40 entries" than as a giant literal initializer.
    inline Font8x8Table BuildFont8x8() {
        Font8x8Table font{};
        auto set = [&](char c, std::initializer_list<uint8_t> rows5) {
            font[static_cast<uint8_t>(c)] = detail::MakeGlyph(rows5);
        };

        set(' ', { 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000 });
        set(':', { 0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000 });
        set('.', { 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100 });
        set('/', { 0b00001, 0b00010, 0b00100, 0b00100, 0b01000, 0b10000, 0b00000 });
        set('%', { 0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011 });
        set('-', { 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000 });

        set('0', { 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110 });
        set('1', { 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 });
        set('2', { 0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111 });
        set('3', { 0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110 });
        set('4', { 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010 });
        set('5', { 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110 });
        set('6', { 0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110 });
        set('7', { 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000 });
        set('8', { 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110 });
        set('9', { 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100 });

        set('A', { 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 });
        set('B', { 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110 });
        set('C', { 0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111 });
        set('D', { 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110 });
        set('E', { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111 });
        set('F', { 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000 });
        set('G', { 0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111 });
        set('H', { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 });
        set('I', { 0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 });
        set('K', { 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001 });
        set('L', { 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111 });
        set('M', { 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001 });
        set('N', { 0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001 });
        set('O', { 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 });
        set('P', { 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000 });
        set('R', { 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001 });
        set('S', { 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110 });
        set('T', { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100 });
        set('U', { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110 });
        set('V', { 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100 });
        set('W', { 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001 });
        set('Y', { 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100 });

        // Add missing uppercase characters J, Q, X, Z
        set('J', { 0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110 });
        set('Q', { 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101 });
        set('X', { 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001 });
        set('Z', { 0b11111, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000, 0b11111 });

        // Add actual lowercase characters for "low", "medium", "high"
        set('d', { 0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111 });
        set('e', { 0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01111 });
        set('g', { 0b00000, 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110 });
        set('h', { 0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001 });
        set('i', { 0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110 });
        set('l', { 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110 });
        set('m', { 0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101 });
        set('o', { 0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110 });
        set('u', { 0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101 });
        set('w', { 0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010 });

        // Fallback mapping for the rest of lowercase characters to uppercase
        for (int c = 'a'; c <= 'z'; ++c) {
            bool allZero = true;
            for (int r = 0; r < 8; ++r) {
                if (font[static_cast<uint8_t>(c)][r] != 0) {
                    allZero = false;
                    break;
                }
            }
            if (allZero) {
                char upper = static_cast<char>(std::toupper(c));
                font[static_cast<uint8_t>(c)] = font[static_cast<uint8_t>(upper)];
            }
        }

        return font;
    }

}

#endif // NDEBUG
