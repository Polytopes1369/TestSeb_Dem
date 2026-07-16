#pragma once
// Low-level asynchronous disk streaming primitive for the virtual geometry cache
// (ClusterFormat.h / CacheFileManager.h sit on top of this for the higher-level .cache format;
// this file has no knowledge of clusters/DAGs -- it only knows how to move 4KB-aligned blocks
// between a file and memory as fast as the storage device allows).
//
// Design: one file handle opened with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (bypasses
// the OS page cache entirely, so read bandwidth reflects the physical device, not RAM), bound to
// a single I/O completion port (IOCP) via CreateIoCompletionPort. A small pool of worker threads
// blocks on GetQueuedCompletionStatus and dispatches each finished read to the caller-supplied
// callback -- this is the standard high-throughput Windows I/O model (used by SQL Server, the
// NTFS/ReFS stack itself, etc.) because it lets the OS keep an arbitrarily deep queue of
// in-flight requests serviced by a small, fixed thread count, instead of one thread per request.
//
// FILE_FLAG_NO_BUFFERING requires every read's file offset, byte count, AND destination buffer
// address to be a multiple of the volume's sector size. This class fixes that granularity at
// kBlockSizeBytes (4096), which is a safe multiple of both common sector sizes (512 and 4096) and
// matches ClusterFormat.h's own kPageSizeBytes, so cluster geometry blocks can be streamed
// through this class with no extra alignment bookkeeping.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace geometry {

    // Block granularity every SubmitRead call must be aligned to (offset, size, and destination
    // buffer address all included) -- see the file header comment.
    constexpr uint32_t kStreamerBlockSizeBytes = 4096u;

    // Low-level, read-only, IOCP-driven async file streamer. Not tied to ClusterFormat.h: any
    // caller needing raw 4KB-aligned block reads at maximum device bandwidth can use this
    // directly (CacheFileManager's higher-level per-cluster API is one such caller).
    class AsyncFileStreamer {
    public:
        // Callback invoked from a worker thread (never the thread that called SubmitRead) once a
        // read completes, successfully or not. `bytesTransferred` is only meaningful when
        // `success` is true.
        using ReadCallback = std::function<void(bool success, uint32_t bytesTransferred)>;

        // `workerThreadCount == 0` auto-selects std::thread::hardware_concurrency() (falling back
        // to 1 if the platform can't report it), matching the IOCP's own concurrency hint so the
        // OS never schedules more threads onto the port than there are cores to run them.
        explicit AsyncFileStreamer(uint32_t workerThreadCount = 0);
        ~AsyncFileStreamer();

        AsyncFileStreamer(const AsyncFileStreamer&) = delete;
        AsyncFileStreamer& operator=(const AsyncFileStreamer&) = delete;

        // Opens `filePath` for unbuffered, overlapped, IOCP-driven reads and starts the worker
        // thread pool. Returns false (logging the OS error) on any failure. Calling Open() on an
        // already-open instance first closes the previous file.
        bool Open(const std::filesystem::path& filePath);

        // Stops the worker threads and closes the file/completion port. Any request submitted
        // before this call but not yet completed will still have its callback invoked (from a
        // worker thread) before the threads are joined -- Close() does not abandon in-flight I/O.
        void Close();

        bool IsOpen() const { return m_FileHandle != INVALID_HANDLE_VALUE; }

        // Submits an asynchronous read of `sizeBytes` at `fileOffset` into `destinationBuffer`.
        // All three of fileOffset, sizeBytes, and the destinationBuffer address must be multiples
        // of kStreamerBlockSizeBytes (destinationBuffer should come from AllocateAlignedBuffer).
        // Returns false immediately (without ever invoking `onComplete`) if the request could not
        // even be launched (bad alignment, closed streamer, or the OS rejected it outright);
        // otherwise `onComplete` is guaranteed to be invoked exactly once, asynchronously, once
        // the read finishes.
        bool SubmitRead(uint64_t fileOffset, void* destinationBuffer, uint32_t sizeBytes, ReadCallback onComplete);

        // Blocks the calling thread until every request submitted so far has had its callback
        // invoked. Safe to call even while other threads (e.g. a completion callback) are still
        // submitting further reads -- it only observes the pending count, so as long as callbacks
        // eventually stop re-submitting, this returns once the queue truly drains.
        void WaitForAllPending();

        uint32_t PendingRequestCount() const { return m_PendingRequests.load(std::memory_order_acquire); }

        // Allocates a buffer whose address is a multiple of `alignmentBytes` (default
        // kStreamerBlockSizeBytes), as FILE_FLAG_NO_BUFFERING requires. Backed by _aligned_malloc
        // (the MSVC CRT's aligned allocator -- the Windows equivalent of POSIX's
        // posix_memalign/aligned_alloc, which are not available on this MSVC/Windows target).
        static void* AllocateAlignedBuffer(size_t sizeBytes, size_t alignmentBytes = kStreamerBlockSizeBytes);
        static void FreeAlignedBuffer(void* buffer);

    private:
        void WorkerThreadMain();
        void OnRequestFinished();

        HANDLE m_FileHandle = INVALID_HANDLE_VALUE;
        HANDLE m_CompletionPort = INVALID_HANDLE_VALUE;
        uint32_t m_WorkerThreadCount = 0;
        std::vector<std::thread> m_WorkerThreads;

        std::atomic<uint32_t> m_PendingRequests{ 0 };
        std::mutex m_PendingMutex;
        std::condition_variable m_PendingCV;
    };

}
