#include "io/BlockCodec.h"

#include <cstring>
#include <vector>

namespace io {

    namespace {

        // Single-slot hash table over 4-byte prefixes (LZ4's own "fast" match-finder strategy):
        // each bucket remembers only the MOST RECENT position seen for that hash, so the finder is
        // O(1) per byte at the cost of missing some older, possibly-longer matches -- a deliberate
        // speed/ratio trade matching this project's non-realtime-critical use (an asset load, not
        // a per-frame hot path).
        constexpr size_t kHashBits = 16;
        constexpr size_t kHashTableSize = size_t{ 1 } << kHashBits;
        constexpr uint32_t kNoPosition = 0xFFFFFFFFu;

        // Fibonacci hashing with LZ4's own well-known multiplicative constant (2654435761, the
        // 32-bit golden-ratio prime) -- a public, widely-reused hash constant, not vendored code.
        uint32_t Hash4(const uint8_t* p) {
            uint32_t v;
            std::memcpy(&v, p, sizeof(v));
            return (v * 2654435761u) >> (32 - kHashBits);
        }

        void EmitExtraLength(uint8_t* dst, size_t& dstPos, size_t remaining) {
            while (remaining >= 255) {
                dst[dstPos++] = 255;
                remaining -= 255;
            }
            dst[dstPos++] = static_cast<uint8_t>(remaining);
        }

        // Reads a length-extension chain starting at src[srcPos], advancing srcPos past it.
        // Returns false (leaving `outExtra` unspecified) if the chain runs past `compressedSize`
        // without terminating -- truncated/corrupt input.
        bool ReadExtraLength(const uint8_t* src, size_t& srcPos, size_t compressedSize, size_t& outExtra) {
            outExtra = 0;
            uint8_t b;
            do {
                if (srcPos >= compressedSize) return false;
                b = src[srcPos++];
                outExtra += b;
            } while (b == 255);
            return true;
        }

    } // namespace

    size_t CompressBlock(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstCapacity) {
        if (dstCapacity < CompressBound(srcSize)) return 0;

        std::vector<uint32_t> hashTable(kHashTableSize, kNoPosition);

        size_t dstPos = 0;
        size_t literalStart = 0;
        size_t srcPos = 0;

        // The last kBlockCodecMinMatch bytes can never start a match search (nothing left to
        // compare against a full min-match length), so they always end up as trailing literals.
        size_t searchLimit = (srcSize >= kBlockCodecMinMatch) ? (srcSize - kBlockCodecMinMatch) : 0;

        while (srcPos < searchLimit) {
            uint32_t hash = Hash4(&src[srcPos]);
            uint32_t candidate = hashTable[hash];
            hashTable[hash] = static_cast<uint32_t>(srcPos);

            size_t matchLen = 0;
            size_t offset = 0;
            if (candidate != kNoPosition) {
                offset = srcPos - candidate;
                if (offset >= 1 && offset <= kBlockCodecMaxOffset &&
                    std::memcmp(&src[candidate], &src[srcPos], kBlockCodecMinMatch) == 0) {
                    size_t maxLen = srcSize - srcPos;
                    while (matchLen < maxLen && src[candidate + matchLen] == src[srcPos + matchLen]) {
                        ++matchLen;
                    }
                }
            }

            if (matchLen < kBlockCodecMinMatch) {
                ++srcPos;
                continue;
            }

            size_t litLen = srcPos - literalStart;
            size_t matchLenEncoded = matchLen - kBlockCodecMinMatch;

            uint8_t litLen4 = static_cast<uint8_t>(litLen < 15 ? litLen : 15);
            uint8_t matchLen4 = static_cast<uint8_t>(matchLenEncoded < 15 ? matchLenEncoded : 15);
            dst[dstPos++] = static_cast<uint8_t>((litLen4 << 4) | matchLen4);

            if (litLen4 == 15) EmitExtraLength(dst, dstPos, litLen - 15);
            if (litLen > 0) {
                std::memcpy(&dst[dstPos], &src[literalStart], litLen);
                dstPos += litLen;
            }

            dst[dstPos++] = static_cast<uint8_t>(offset & 0xFFu);
            dst[dstPos++] = static_cast<uint8_t>((offset >> 8) & 0xFFu);
            if (matchLen4 == 15) EmitExtraLength(dst, dstPos, matchLenEncoded - 15);

            srcPos += matchLen;
            literalStart = srcPos;
        }

        // Final sequence: every remaining byte is a literal, with no trailing match (see
        // BlockCodec.h / DecompressBlock's own comment on how the decoder recognizes this case
        // purely from having produced `decompressedSize` bytes, with no dedicated format flag).
        size_t finalLitLen = srcSize - literalStart;
        uint8_t finalLitLen4 = static_cast<uint8_t>(finalLitLen < 15 ? finalLitLen : 15);
        dst[dstPos++] = static_cast<uint8_t>(finalLitLen4 << 4); // Low nibble (match length) unused for the final, match-less sequence.
        if (finalLitLen4 == 15) EmitExtraLength(dst, dstPos, finalLitLen - 15);
        if (finalLitLen > 0) {
            std::memcpy(&dst[dstPos], &src[literalStart], finalLitLen);
            dstPos += finalLitLen;
        }

        return dstPos;
    }

    size_t DecompressBlock(const uint8_t* src, size_t compressedSize, uint8_t* dst, size_t decompressedSize) {
        size_t srcPos = 0;
        size_t dstPos = 0;

        while (dstPos < decompressedSize) {
            if (srcPos >= compressedSize) return 0; // Truncated: a token was expected but the stream ended.

            uint8_t token = src[srcPos++];
            size_t litLen = token >> 4;
            size_t matchLen4 = token & 0x0Fu;

            if (litLen == 15) {
                size_t extra = 0;
                if (!ReadExtraLength(src, srcPos, compressedSize, extra)) return 0;
                litLen += extra;
            }

            if (srcPos + litLen > compressedSize || dstPos + litLen > decompressedSize) return 0; // Corrupt: literal run overruns a buffer.
            std::memcpy(&dst[dstPos], &src[srcPos], litLen);
            srcPos += litLen;
            dstPos += litLen;

            if (dstPos == decompressedSize) break; // Final sequence reached exactly: no match follows (see CompressBlock's matching comment).

            if (srcPos + 2 > compressedSize) return 0;
            uint32_t offset = static_cast<uint32_t>(src[srcPos]) | (static_cast<uint32_t>(src[srcPos + 1]) << 8);
            srcPos += 2;
            if (offset == 0 || offset > dstPos) return 0; // Back-reference points before the start of the output -- corrupt.

            size_t matchLen = matchLen4;
            if (matchLen4 == 15) {
                size_t extra = 0;
                if (!ReadExtraLength(src, srcPos, compressedSize, extra)) return 0;
                matchLen += extra;
            }
            matchLen += kBlockCodecMinMatch;

            if (dstPos + matchLen > decompressedSize) return 0; // Corrupt: match would overrun the output buffer.

            size_t matchSrc = dstPos - offset;
            // Byte-by-byte, NOT memcpy: offset < matchLen is legal and expected (it is exactly how
            // this codec encodes a run of a repeated short pattern, e.g. offset=1 repeats a single
            // byte matchLen times) -- memcpy's behavior on overlapping regions is undefined, so an
            // overlap-safe forward copy is required here.
            for (size_t i = 0; i < matchLen; ++i) {
                dst[dstPos + i] = dst[matchSrc + i];
            }
            dstPos += matchLen;
        }

        return dstPos;
    }

}
