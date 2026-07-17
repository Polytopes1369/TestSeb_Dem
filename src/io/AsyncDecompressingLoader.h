#pragma once
// Full non-blocking "read compressed chunk from disk, decompress, notify" pipeline, built by
// composing two already-existing, already-tested primitives rather than reimplementing either:
//   - geometry::AsyncFileStreamer: the IOCP-driven, FILE_FLAG_NO_BUFFERING async disk reader (see
//     AsyncFileStreamer.h) -- unchanged, reused exactly as CacheFileManager already reuses it for
//     virtual-geometry cluster paging.
//   - core::LoadingManager: the generic CPU-bound background-work pool (see LoadingManager.h) --
//     decompression is CPU work, and LoadingManager already exists specifically so CPU-bound jobs
//     have somewhere to run other than the IOCP thread or the main thread.
//
// Why decompression must NOT run on the AsyncFileStreamer's own IOCP worker thread: that pool is
// sized to keep the completion port serviced (dispatching finished reads promptly so the next
// queued read can be issued) -- blocking one of its few threads on a CPU-bound decompression pass
// starves every OTHER in-flight read's completion dispatch behind it, defeating the whole point
// of using an IOCP in the first place. So the read-completion callback below (running on an
// AsyncFileStreamer worker thread) does the absolute minimum -- check the read succeeded, then
// immediately hand the actual decompression off to core::LoadingManager's separate pool.
//
// Threading summary for one SubmitChunkLoad() call:
//   caller thread -> AsyncFileStreamer (IOCP worker thread reads disk) -> core::LoadingManager
//   (a DIFFERENT worker thread decompresses) -> ChunkLoadCallback (only ever invoked from whatever
//   thread calls `workerPool`'s own PumpCompletions(), never directly from either worker pool).
//   AsyncDecompressingLoader does NOT expose its own PumpCompletions wrapper: `workerPool` is
//   meant to be one shared instance every CPU-bound system in the engine rides on, so the caller's
//   main loop already has exactly one PumpCompletions() call site draining all of them -- adding a
//   second, per-subsystem pump method here would just invite an ambiguous "which one do I call"
//   footgun for no benefit.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>

#include "core/LoadingManager.h"
#include "io/AsyncFileStreamer.h"

namespace io {

    // Describes one compressed chunk to load. All three offset/size fields describing the DISK
    // read (fileOffset, alignedReadSizeBytes) must respect geometry::kStreamerBlockSizeBytes
    // alignment (see AsyncFileStreamer::SubmitRead) -- the true compressed payload
    // (compressedSizeBytes) is normally smaller than alignedReadSizeBytes, with the remainder
    // being padding bytes the decompressor never reads. A caller typically gets all 4 fields from
    // an index/table elsewhere (e.g. an OFPA actor's compressed-chunk directory), exactly the way
    // CacheFileManager's ClusterIndexEntry already supplies virtualAddress/blockSizeBytes for its
    // own paged reads.
    struct ChunkLoadRequest {
        uint64_t fileOffset = 0;
        uint32_t alignedReadSizeBytes = 0;
        uint32_t compressedSizeBytes = 0;
        uint32_t decompressedSizeBytes = 0;
    };

    // Invoked from whichever thread calls AsyncDecompressingLoader::PumpCompletions -- NEVER
    // directly from an AsyncFileStreamer or core::LoadingManager worker thread (see class
    // comment), so it is exactly as safe to touch non-thread-safe engine/Vulkan state here as from
    // that PumpCompletions() call site itself. `success` is false if the disk read failed (short
    // read, I/O error) or the decompressed bytes failed to validate (io::DecompressBlock detected
    // corruption/truncation) -- `decompressedData` is null and `decompressedSizeBytes` is 0 in
    // that case.
    using ChunkLoadCallback = std::function<void(bool success, std::unique_ptr<uint8_t[]> decompressedData, uint32_t decompressedSizeBytes)>;

    class AsyncDecompressingLoader {
    public:
        // `workerPool` is owned by the caller and must outlive this instance -- typically the same
        // shared core::LoadingManager instance every other CPU-bound background system in the
        // engine uses (see that class's own header comment on why it is deliberately generic
        // rather than a dedicated decompression-only pool).
        explicit AsyncDecompressingLoader(core::LoadingManager& workerPool, uint32_t ioWorkerThreadCount = 0);
        ~AsyncDecompressingLoader();

        AsyncDecompressingLoader(const AsyncDecompressingLoader&) = delete;
        AsyncDecompressingLoader& operator=(const AsyncDecompressingLoader&) = delete;

        bool Open(const std::filesystem::path& filePath);
        void Close();
        bool IsOpen() const;

        // Never blocks. Returns false immediately (without ever invoking `onComplete`) if the
        // request could not even be launched (bad alignment, closed loader, or the aligned read
        // buffer failed to allocate) -- matching AsyncFileStreamer::SubmitRead's own contract.
        bool SubmitChunkLoad(const ChunkLoadRequest& request, ChunkLoadCallback onComplete);

    private:
        geometry::AsyncFileStreamer m_Streamer;
        core::LoadingManager& m_WorkerPool;
    };

}
