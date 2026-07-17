// Standalone, framework-free unit test for io::BlockCodec (src/io/BlockCodec.h): round-trips a
// variety of payloads (empty, tiny, highly repetitive, random/incompressible, self-overlapping
// runs, long literal runs needing length-extension bytes) through CompressBlock/DecompressBlock,
// and verifies corruption is detected rather than causing an out-of-bounds read/write. Exits 0 if
// every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching this project's existing tests/*.cpp convention.

#include "io/BlockCodec.h"

#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    void RoundTripCheck(const std::string& label, const std::vector<uint8_t>& original) {
        std::vector<uint8_t> compressed(io::CompressBound(original.size()));
        size_t compressedSize = io::CompressBlock(original.data(), original.size(), compressed.data(), compressed.size());
        Check(compressedSize > 0, label + ": CompressBlock should always succeed (produce at least the terminating token byte) given a CompressBound-sized destination");

        std::vector<uint8_t> decompressed(original.size());
        size_t decompressedSize = io::DecompressBlock(compressed.data(), compressedSize, decompressed.data(), original.size());

        Check(decompressedSize == original.size(), label + ": DecompressBlock should reproduce the exact original size");
        Check(decompressed == original, label + ": DecompressBlock should reproduce byte-identical content");
    }

    void TestEmpty() {
        RoundTripCheck("empty", {});
    }

    void TestTinyIncompressible() {
        RoundTripCheck("tiny", { 0x01, 0x02, 0x03 });
    }

    void TestHighlyRepetitive() {
        // A long run of a single repeated byte: forces a self-overlapping match (offset=1, a
        // match length far greater than the offset) through DecompressBlock's overlap-safe copy.
        std::vector<uint8_t> data(10000, 0x7Au);
        RoundTripCheck("highly repetitive (RLE-style)", data);
    }

    void TestRepeatingPattern() {
        // A repeating multi-byte pattern: exercises matches whose offset equals the pattern
        // length, still with offset < matchLen once the match runs for multiple pattern periods.
        std::vector<uint8_t> data;
        const std::string pattern = "AliceInWonderland";
        for (int i = 0; i < 2000; ++i) {
            data.insert(data.end(), pattern.begin(), pattern.end());
        }
        RoundTripCheck("repeating multi-byte pattern", data);
    }

    void TestRandomIncompressible() {
        std::mt19937 rng(12345u);
        std::uniform_int_distribution<int> dist(0, 255);
        std::vector<uint8_t> data(5000);
        for (uint8_t& b : data) b = static_cast<uint8_t>(dist(rng));
        RoundTripCheck("random incompressible", data);
    }

    void TestLongLiteralRunNeedingExtensionBytes() {
        // >= 15 leading literal bytes before the first match forces the litLen4==15 extension-byte
        // encoding path in both CompressBlock and DecompressBlock.
        std::vector<uint8_t> data;
        for (int i = 0; i < 500; ++i) data.push_back(static_cast<uint8_t>(i & 0xFF)); // Non-repeating: no match found here.
        for (int i = 0; i < 20; ++i) { data.push_back('X'); data.push_back('Y'); data.push_back('Z'); data.push_back('W'); } // Then a repeated 4-byte pattern to force at least one match.
        RoundTripCheck("long literal run + extension bytes", data);
    }

    void TestMixedRealistic() {
        // A blend of structured (repeated struct-like records) and varying data, closer to what a
        // real actor/geometry chunk would look like.
        std::vector<uint8_t> data;
        std::mt19937 rng(777u);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int record = 0; record < 200; ++record) {
            for (int i = 0; i < 16; ++i) data.push_back(static_cast<uint8_t>('R')); // Repeated header-like prefix.
            for (int i = 0; i < 8; ++i) data.push_back(static_cast<uint8_t>(dist(rng))); // Varying payload.
        }
        RoundTripCheck("mixed structured+random", data);
    }

    void TestCorruptionIsDetected() {
        std::vector<uint8_t> original = { 'A','A','A','A','A','A','A','A','A','A','B','C','D' };
        std::vector<uint8_t> compressed(io::CompressBound(original.size()));
        size_t compressedSize = io::CompressBlock(original.data(), original.size(), compressed.data(), compressed.size());
        Check(compressedSize > 0, "corruption test setup: CompressBlock should succeed");

        // Truncate the compressed stream by half: DecompressBlock must fail cleanly (return 0),
        // never read/write out of bounds (this test relies on running cleanly under a normal
        // debug allocator/heap guard to catch any OOB access as a crash rather than silently).
        std::vector<uint8_t> decompressed(original.size(), 0xEE);
        size_t truncatedSize = compressedSize / 2;
        size_t result = io::DecompressBlock(compressed.data(), truncatedSize, decompressed.data(), original.size());
        Check(result == 0, "DecompressBlock should return 0 for a truncated/corrupt compressed stream");
    }

    void TestCompressBoundIsSufficient() {
        for (size_t size : { size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{255}, size_t{256}, size_t{1000000} }) {
            Check(io::CompressBound(size) >= size, "CompressBound must always be >= the source size for size=" + std::to_string(size));
        }
    }

}

int main() {
    TestEmpty();
    TestTinyIncompressible();
    TestHighlyRepetitive();
    TestRepeatingPattern();
    TestRandomIncompressible();
    TestLongLiteralRunNeedingExtensionBytes();
    TestMixedRealistic();
    TestCorruptionIsDetected();
    TestCompressBoundIsSufficient();

    if (g_failCount == 0) {
        std::cout << "[PASS] All BlockCodec checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
