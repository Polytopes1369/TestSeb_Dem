// Standalone correctness check + bandwidth benchmark for geometry::AsyncFileStreamer
// (src/geometry/AsyncFileStreamer.h/.cpp). No Vulkan/GLFW dependency.
//
// What it does:
//   1. Writes a deterministic, per-block-verifiable benchmark file via a plain buffered write.
//   2. Reads the whole file back through AsyncFileStreamer at queue depth 1 and verifies every
//      block's content matches what was written -- correctness first, exercising the full
//      FILE_FLAG_NO_BUFFERING + IOCP + aligned-buffer path end-to-end.
//   3. Re-reads the whole file at increasing queue depths (4, 16, 64, 128), reporting achieved
//      bandwidth (MB/s) for each -- demonstrating how a deeper overlapped queue approaches the
//      storage device's real maximum read bandwidth, entirely bypassing the OS page cache.
//
// Exits 0 if every correctness check passes (queue-depth scaling is informational, not a
// pass/fail condition -- achievable bandwidth is hardware-dependent), non-zero otherwise, so it
// can be registered with CTest without pulling in any external test/benchmark framework.

#include "geometry/AsyncFileStreamer.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

    // 128 MB: large enough that queue-depth scaling is clearly visible on real storage, small
    // enough that the multiple full-file passes below stay fast on any CI machine.
    constexpr uint64_t kBenchmarkFileSizeBytes = 128ull * 1024ull * 1024ull;
    constexpr uint64_t kBenchmarkBlockCount = kBenchmarkFileSizeBytes / geometry::kStreamerBlockSizeBytes;
    constexpr uint32_t kWordsPerBlock = geometry::kStreamerBlockSizeBytes / sizeof(uint32_t);

    // Deterministic, cheap-to-regenerate content for block `blockIndex`, word `wordIndex` --
    // lets every block be verified against a formula instead of keeping the whole file in RAM.
    uint32_t ExpectedWord(uint64_t blockIndex, uint32_t wordIndex) {
        return static_cast<uint32_t>(blockIndex) * 2654435761u + wordIndex * 40503u + 1u;
    }

    void FillBlockPattern(uint64_t blockIndex, uint8_t* block) {
        uint32_t* words = reinterpret_cast<uint32_t*>(block);
        for (uint32_t w = 0; w < kWordsPerBlock; ++w) {
            words[w] = ExpectedWord(blockIndex, w);
        }
    }

    bool VerifyBlockPattern(uint64_t blockIndex, const uint8_t* block) {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(block);
        for (uint32_t w = 0; w < kWordsPerBlock; ++w) {
            if (words[w] != ExpectedWord(blockIndex, w)) {
                return false;
            }
        }
        return true;
    }

    // Plain buffered write: FILE_FLAG_NO_BUFFERING only needs to apply to the read side under
    // test, and using it for the write here as well is unnecessary.
    bool WriteBenchmarkFile(const std::filesystem::path& filePath) {
        std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
        if (!file) {
            std::cerr << "[AsyncFileStreamerBenchmark] Failed to create '" << filePath.string() << "'.\n";
            return false;
        }
        std::vector<uint8_t> block(geometry::kStreamerBlockSizeBytes);
        for (uint64_t b = 0; b < kBenchmarkBlockCount; ++b) {
            FillBlockPattern(b, block.data());
            file.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
        }
        return file.good();
    }

    // Shared state for one queue-depth pass: `queueDepth` reusable aligned buffers, each
    // continuously refilled with the next not-yet-claimed block until the file is exhausted.
    struct BenchmarkPass {
        std::atomic<uint64_t> nextBlockIndex{ 0 };
        std::atomic<uint64_t> totalBytesRead{ 0 };
        std::atomic<uint64_t> mismatchCount{ 0 };
        geometry::AsyncFileStreamer* streamer = nullptr;
        bool verify = false;
    };

    void SubmitNextBlock(BenchmarkPass& pass, uint8_t* buffer) {
        uint64_t blockIndex = pass.nextBlockIndex.fetch_add(1, std::memory_order_relaxed);
        if (blockIndex >= kBenchmarkBlockCount) {
            return; // File exhausted: this buffer slot goes idle for the remainder of the pass.
        }
        uint64_t offset = blockIndex * geometry::kStreamerBlockSizeBytes;
        pass.streamer->SubmitRead(offset, buffer, geometry::kStreamerBlockSizeBytes,
            [&pass, buffer, blockIndex](bool success, uint32_t bytesTransferred) {
                if (success) {
                    pass.totalBytesRead.fetch_add(bytesTransferred, std::memory_order_relaxed);
                    if (pass.verify && !VerifyBlockPattern(blockIndex, buffer)) {
                        pass.mismatchCount.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                SubmitNextBlock(pass, buffer); // Refill this slot with the next unclaimed block.
                });
    }

    // Runs one full-file read pass at the given queue depth and returns the achieved bandwidth
    // in MB/s. `outMismatches` is only meaningful when `verify` is true.
    double RunPass(geometry::AsyncFileStreamer& streamer, uint32_t queueDepth, bool verify, uint64_t& outMismatches) {
        std::vector<uint8_t*> buffers(queueDepth);
        for (uint8_t*& b : buffers) {
            b = static_cast<uint8_t*>(geometry::AsyncFileStreamer::AllocateAlignedBuffer(geometry::kStreamerBlockSizeBytes));
        }

        BenchmarkPass pass;
        pass.streamer = &streamer;
        pass.verify = verify;

        auto start = std::chrono::steady_clock::now();
        for (uint8_t* buffer : buffers) {
            SubmitNextBlock(pass, buffer);
        }
        streamer.WaitForAllPending();
        auto end = std::chrono::steady_clock::now();

        for (uint8_t* buffer : buffers) {
            geometry::AsyncFileStreamer::FreeAlignedBuffer(buffer);
        }

        outMismatches = pass.mismatchCount.load();
        double elapsedSeconds = std::chrono::duration<double>(end - start).count();
        double megabytesRead = static_cast<double>(pass.totalBytesRead.load()) / (1024.0 * 1024.0);
        return (elapsedSeconds > 0.0) ? megabytesRead / elapsedSeconds : 0.0;
    }

} // namespace

int main() {
    std::filesystem::path filePath = "async_streamer_benchmark.tmp";
    int failCount = 0;

    std::cout << "[AsyncFileStreamerBenchmark] Writing " << (kBenchmarkFileSizeBytes / (1024 * 1024))
        << " MB benchmark file (" << kBenchmarkBlockCount << " blocks)...\n";
    if (!WriteBenchmarkFile(filePath)) {
        std::cerr << "[FAIL] Could not create the benchmark file.\n";
        return 1;
    }

    // Pass 1: queue depth 1, content-verified -- proves correctness of the full
    // FILE_FLAG_NO_BUFFERING + IOCP + aligned-buffer path before trusting any bandwidth number.
    {
        geometry::AsyncFileStreamer streamer;
        if (!streamer.Open(filePath)) {
            std::cerr << "[FAIL] AsyncFileStreamer::Open failed.\n";
            ++failCount;
        }
        else {
            uint64_t mismatches = 0;
            double mbPerSec = RunPass(streamer, 1, /*verify=*/true, mismatches);
            std::cout << "[AsyncFileStreamerBenchmark] Queue depth   1: " << mbPerSec << " MB/s (content-verified, "
                << mismatches << " mismatch(es)).\n";
            if (mismatches != 0) {
                std::cerr << "[FAIL] Data corruption detected while reading back through AsyncFileStreamer.\n";
                ++failCount;
            }
        }
    }

    // Passes 2+: increasing queue depth, throughput-focused -- demonstrates how a deeper
    // in-flight queue approaches the storage device's real maximum unbuffered read bandwidth.
    for (uint32_t queueDepth : { 4u, 16u, 64u, 128u }) {
        geometry::AsyncFileStreamer streamer;
        if (!streamer.Open(filePath)) {
            std::cerr << "[FAIL] AsyncFileStreamer::Open failed for queue depth " << queueDepth << ".\n";
            ++failCount;
            continue;
        }
        uint64_t mismatches = 0;
        double mbPerSec = RunPass(streamer, queueDepth, /*verify=*/false, mismatches);
        std::cout << "[AsyncFileStreamerBenchmark] Queue depth " << queueDepth << ": " << mbPerSec << " MB/s.\n";
    }

    std::error_code removeEc;
    std::filesystem::remove(filePath, removeEc);

    if (failCount == 0) {
        std::cout << "[AsyncFileStreamerBenchmark] All correctness checks PASSED.\n";
        return 0;
    }
    std::cerr << "[AsyncFileStreamerBenchmark] " << failCount << " check(s) FAILED.\n";
    return 1;
}
