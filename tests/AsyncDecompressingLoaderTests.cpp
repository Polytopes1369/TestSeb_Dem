// Standalone, framework-free unit test for io::AsyncDecompressingLoader (composing
// geometry::AsyncFileStreamer's real IOCP async disk I/O with core::LoadingManager's real worker
// pool and io::BlockCodec's real compressor): writes a compressed, block-padded chunk to a temp
// file on disk, then drives the full non-blocking read-then-decompress-then-notify pipeline and
// verifies the decompressed bytes come back identical -- exercising actual OS-level asynchronous
// I/O and cross-thread handoff, not a mock. Exits 0 if every check passes, non-zero otherwise --
// registered with CTest (see the top-level CMakeLists.txt), matching this project's existing
// tests/*.cpp convention.
//
// Draining polls core::LoadingManager::PumpCompletions in a bounded loop, yielding the calling
// thread between attempts: a tight, non-yielding spin loop can starve the IOCP/worker threads of
// CPU time on a machine with few cores (the polling thread never actually blocks, so the OS
// scheduler has no reason to preempt it in favor of the threads doing the real work), which was
// observed to make this test hang for thousands of iterations despite the underlying pipeline
// completing in well under a millisecond -- see AsyncDecompressingLoader.cpp's own threading
// model comment for why the real work always happens off this polling thread.

#include "io/AsyncDecompressingLoader.h"
#include "io/AsyncFileStreamer.h"
#include "io/BlockCodec.h"
#include "core/LoadingManager.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <thread>
#include <fstream>
#include <iostream>
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

    // Compresses `original`, pads the compressed bytes up to a multiple of
    // geometry::kStreamerBlockSizeBytes with zeros (FILE_FLAG_NO_BUFFERING's alignment
    // requirement, see AsyncFileStreamer.h), and writes the result to `filePath`. Returns the
    // request describing exactly how to load it back. Ordinary buffered std::ofstream is fine
    // here -- the alignment requirement only applies to AsyncFileStreamer's own unbuffered READS,
    // never to how a test fixture is written.
    io::ChunkLoadRequest WriteCompressedChunkFile(const std::filesystem::path& filePath, const std::vector<uint8_t>& original) {
        std::vector<uint8_t> compressed(io::CompressBound(original.size()));
        size_t compressedSize = io::CompressBlock(original.data(), original.size(), compressed.data(), compressed.size());

        size_t alignedSize = ((compressedSize + geometry::kStreamerBlockSizeBytes - 1) / geometry::kStreamerBlockSizeBytes) * geometry::kStreamerBlockSizeBytes;
        if (alignedSize == 0) alignedSize = geometry::kStreamerBlockSizeBytes;
        std::vector<uint8_t> padded(alignedSize, 0);
        std::memcpy(padded.data(), compressed.data(), compressedSize);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(padded.data()), static_cast<std::streamsize>(padded.size()));
        out.close();

        io::ChunkLoadRequest request;
        request.fileOffset = 0;
        request.alignedReadSizeBytes = static_cast<uint32_t>(alignedSize);
        request.compressedSizeBytes = static_cast<uint32_t>(compressedSize);
        request.decompressedSizeBytes = static_cast<uint32_t>(original.size());
        return request;
    }

    void TestSuccessfulRoundTrip() {
        std::vector<uint8_t> original;
        const std::string text = "The quick brown fox jumps over the lazy dog. ";
        for (int i = 0; i < 300; ++i) original.insert(original.end(), text.begin(), text.end());

        std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "AsyncDecompressingLoaderTests";
        std::error_code ec;
        std::filesystem::create_directories(scratchDir, ec);
        std::filesystem::path filePath = scratchDir / "chunk_ok.bin";

        io::ChunkLoadRequest request = WriteCompressedChunkFile(filePath, original);

        core::LoadingManager pool(2);
        io::AsyncDecompressingLoader loader(pool, 2);
        Check(loader.Open(filePath), "AsyncDecompressingLoader::Open should succeed for a file that was just written");

        std::atomic<bool> done{ false };
        bool success = false;
        std::vector<uint8_t> received;

        bool launched = loader.SubmitChunkLoad(request, [&](bool ok, std::unique_ptr<uint8_t[]> data, uint32_t size) {
            success = ok;
            if (ok && data) received.assign(data.get(), data.get() + size);
            done.store(true, std::memory_order_release);
            });
        Check(launched, "SubmitChunkLoad should launch successfully for a well-formed, aligned request");

        for (int i = 0; i < 2000 && !done.load(std::memory_order_acquire); ++i) {
            pool.PumpCompletions(8);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        Check(done.load(std::memory_order_acquire), "the chunk load should have completed within the poll bound");
        Check(success, "the chunk load should report success for an uncorrupted chunk");
        Check(received == original, "the decompressed bytes should be byte-identical to the original payload");
    }

    void TestCorruptedChunkReportsFailure() {
        std::vector<uint8_t> original(2000, 0x42u);
        std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "AsyncDecompressingLoaderTests";
        std::error_code ec;
        std::filesystem::create_directories(scratchDir, ec);
        std::filesystem::path filePath = scratchDir / "chunk_corrupt.bin";

        io::ChunkLoadRequest request = WriteCompressedChunkFile(filePath, original);
        // Lie about the compressed size (much smaller than what was actually written): forces
        // io::DecompressBlock to see a truncated stream and return failure, exercising the
        // pipeline's corruption-propagation path end to end.
        request.compressedSizeBytes = request.compressedSizeBytes > 4 ? 4u : request.compressedSizeBytes;

        core::LoadingManager pool(2);
        io::AsyncDecompressingLoader loader(pool, 2);
        Check(loader.Open(filePath), "AsyncDecompressingLoader::Open should succeed for a file that was just written");

        std::atomic<bool> done{ false };
        bool success = true; // Default true so a callback that never runs would visibly fail the check below.
        bool dataWasNull = false;

        bool launched = loader.SubmitChunkLoad(request, [&](bool ok, std::unique_ptr<uint8_t[]> data, uint32_t) {
            success = ok;
            dataWasNull = (data == nullptr);
            done.store(true, std::memory_order_release);
            });
        Check(launched, "SubmitChunkLoad should still launch (the corruption is only discovered once decompression runs)");

        for (int i = 0; i < 2000 && !done.load(std::memory_order_acquire); ++i) {
            pool.PumpCompletions(8);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        Check(done.load(std::memory_order_acquire), "the corrupted chunk load should still complete (with failure) within the poll bound");
        Check(!success, "a truncated/corrupted compressed chunk must report failure, not silently succeed");
        Check(dataWasNull, "a failed chunk load must hand back a null data pointer");
    }

    void TestMisalignedRequestRejectedImmediately() {
        std::vector<uint8_t> original(1000, 0x11u);
        std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "AsyncDecompressingLoaderTests";
        std::error_code ec;
        std::filesystem::create_directories(scratchDir, ec);
        std::filesystem::path filePath = scratchDir / "chunk_misaligned.bin";

        io::ChunkLoadRequest request = WriteCompressedChunkFile(filePath, original);
        request.fileOffset = 1; // Not a multiple of geometry::kStreamerBlockSizeBytes -- must be rejected up front.

        core::LoadingManager pool(1);
        io::AsyncDecompressingLoader loader(pool, 1);
        Check(loader.Open(filePath), "AsyncDecompressingLoader::Open should succeed for a file that was just written");

        bool callbackInvoked = false;
        bool launched = loader.SubmitChunkLoad(request, [&](bool, std::unique_ptr<uint8_t[]>, uint32_t) {
            callbackInvoked = true;
            });

        Check(!launched, "SubmitChunkLoad must return false immediately for a misaligned fileOffset");

        pool.WaitIdle();
        pool.PumpCompletions(8);
        Check(!callbackInvoked, "a rejected (unaligned) request must never invoke its callback, immediately or later");
    }

    // Regression test for a real shutdown-race bug in geometry::AsyncFileStreamer::Close(): the
    // documented contract (AsyncFileStreamer.h) promises every request submitted before Close()
    // still gets its callback invoked, from a worker thread, before the threads are joined. The
    // original implementation posted the IOCP shutdown packets BEFORE draining pending requests,
    // so a worker thread could dequeue its (no-I/O-required) shutdown packet and exit while an
    // earlier real ReadFile on the same file was still genuinely in flight on the device --
    // leaking that request's heap-allocated StreamingRequest and silently dropping its callback.
    // Exercises geometry::AsyncFileStreamer directly (not through AsyncDecompressingLoader) since
    // this is purely about the low-level Close()/SubmitRead race, not decompression.
    void TestCloseDrainsInFlightReads() {
        // Many small reads over a reasonably large file, submitted back-to-back with NO pumping,
        // sleeping, or waiting in between -- maximizes the chance several are still genuinely
        // in-flight on the device at the moment Close() is called immediately afterward. Even on
        // a run where every read happens to finish before Close() is reached (fast NVMe, warm
        // cache, etc.), the assertions below still hold Close() to the invariant it must uphold
        // either way: exactly one callback per successfully-launched SubmitRead, observed before
        // Close() returns -- so this test both tries to catch the race directly and acts as a
        // standing regression guard against the ordering being reintroduced.
        constexpr uint32_t kBlockCount = 4096; // 16 MB at kStreamerBlockSizeBytes (4096 B) each.

        std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "AsyncDecompressingLoaderTests";
        std::error_code ec;
        std::filesystem::create_directories(scratchDir, ec);
        std::filesystem::path filePath = scratchDir / "close_drain.bin";

        {
            std::vector<uint8_t> block(geometry::kStreamerBlockSizeBytes, 0xABu);
            std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
            for (uint32_t b = 0; b < kBlockCount; ++b) {
                out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
            }
        }

        geometry::AsyncFileStreamer streamer;
        Check(streamer.Open(filePath), "AsyncFileStreamer::Open should succeed for a file that was just written");

        std::vector<void*> buffers(kBlockCount, nullptr);
        std::atomic<uint32_t> launchedCount{ 0 };
        std::atomic<uint32_t> callbackCount{ 0 };
        std::atomic<uint32_t> successCount{ 0 };

        for (uint32_t b = 0; b < kBlockCount; ++b) {
            buffers[b] = geometry::AsyncFileStreamer::AllocateAlignedBuffer(geometry::kStreamerBlockSizeBytes);
            bool launched = streamer.SubmitRead(
                static_cast<uint64_t>(b) * geometry::kStreamerBlockSizeBytes,
                buffers[b], geometry::kStreamerBlockSizeBytes,
                [&callbackCount, &successCount](bool success, uint32_t) {
                    callbackCount.fetch_add(1, std::memory_order_relaxed);
                    if (success) successCount.fetch_add(1, std::memory_order_relaxed);
                });
            if (launched) launchedCount.fetch_add(1, std::memory_order_relaxed);
        }

        // The call under test: no drain performed by this test itself beforehand, on purpose --
        // Close() alone is responsible for making this safe.
        streamer.Close();

        Check(callbackCount.load() == launchedCount.load(),
            "every successfully-launched SubmitRead's callback must have been invoked by the time Close() returns "
            "(a mismatch means Close() abandoned in-flight I/O, dropping callbacks and leaking StreamingRequests)");
        Check(successCount.load() == launchedCount.load(),
            "every successfully-launched read of this freshly-written, unmodified file should complete successfully");

        for (void* buffer : buffers) {
            geometry::AsyncFileStreamer::FreeAlignedBuffer(buffer);
        }
    }

}

int main() {
    TestSuccessfulRoundTrip();
    TestCorruptedChunkReportsFailure();
    TestMisalignedRequestRejectedImmediately();
    TestCloseDrainsInFlightReads();

    if (g_failCount == 0) {
        std::cout << "[PASS] All AsyncDecompressingLoader checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
