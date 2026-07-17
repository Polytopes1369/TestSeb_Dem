#pragma once
// This project's lightweight LZ77-family block compressor/decompressor: a hash-chained greedy
// match finder over a 4-byte minimum match, encoded as a token stream of
// (literalRun, backReference) sequences -- the same structural family real-world block codecs
// like LZ4 use (nibble-packed token byte + overflow-extension bytes for long runs, a 2-byte
// back-reference offset, self-overlapping copies for run-length-style repeats).
//
// NOT a vendored copy of liblz4/libzstd: this codebase's build environment has no network access
// to fetch and vet a third-party library source drop, so this is the project's OWN, fully
// self-contained implementation (both the encoder and decoder are written and tested together
// here) rather than an approximation of one. It is deliberately API-shaped like a real block
// codec's C API (CompressBlock/DecompressBlock/CompressBound, mirroring
// LZ4_compress_default/LZ4_decompress_safe/LZ4_compressBound's signatures) specifically so that
// vendoring the real liblz4 or libzstd later is a drop-in replacement at every call site in
// AsyncDecompressingLoader.cpp, not a redesign.

#include <cstddef>
#include <cstdint>

namespace io {

    // Minimum back-reference match length this codec ever emits -- a match shorter than this
    // would cost more to encode (token + offset bytes) than it saves versus literal bytes.
    inline constexpr size_t kBlockCodecMinMatch = 4;

    // Maximum back-reference distance (2-byte offset field): a match can only reference a byte up
    // to this far behind the current output position.
    inline constexpr size_t kBlockCodecMaxOffset = 65535;

    // Worst-case compressed size for a `srcSize`-byte input (all-literals case, plus token and
    // length-extension overhead) -- size `dst` to at least this before calling CompressBlock.
    // Mirrors LZ4_compressBound's own formula/contract (always >= srcSize, cheap to compute
    // up front so callers never need a resize-and-retry loop).
    constexpr size_t CompressBound(size_t srcSize) {
        return srcSize + (srcSize / 255) + 16;
    }

    // Compresses `srcSize` bytes at `src` into `dst` (capacity `dstCapacity`, which must be >=
    // CompressBound(srcSize)). Returns the number of bytes written to `dst`, or 0 if `dstCapacity`
    // is too small (this codec never partially writes a corrupt/truncated stream on failure).
    size_t CompressBlock(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t dstCapacity);

    // Decompresses `compressedSize` bytes at `src` (a buffer previously produced by
    // CompressBlock) into `dst`, which must be exactly `decompressedSize` bytes -- the caller must
    // already know the original size out of band (this codec's block format is not
    // self-describing about it, matching real block-codec APIs like LZ4's own: the decompressed
    // size is recorded in whatever container format wraps the block, e.g. ChunkLoadRequest in
    // AsyncDecompressingLoader.h, never inside the block itself).
    //
    // Returns `decompressedSize` on success. Returns 0 on ANY detected corruption or truncation
    // (a token whose literal run or match would read/write past a buffer's end, an out-of-range
    // back-reference offset, or a `src` that runs out before `decompressedSize` bytes have been
    // produced) -- this codec never writes past `dst`+`decompressedSize` or reads past
    // `src`+`compressedSize`, even for adversarial/corrupt input.
    size_t DecompressBlock(const uint8_t* src, size_t compressedSize, uint8_t* dst, size_t decompressedSize);

}
