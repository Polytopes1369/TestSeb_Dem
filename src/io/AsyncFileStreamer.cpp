#include "io/AsyncFileStreamer.h"
#include "core/Logger.h"
#include "core/ThreadingUtil.h"

#include <format>

namespace geometry {

    namespace {

        // Sentinel completion key posted by Close() to wake each worker thread out of its
        // blocking GetQueuedCompletionStatus call. Real read completions always report the
        // completion key associated with the file handle at CreateIoCompletionPort time (0, see
        // AsyncFileStreamer::Open), so this value can never collide with a genuine completion.
        constexpr ULONG_PTR kShutdownCompletionKey = 1;

        // One request's context: OVERLAPPED MUST be the first member. The Win32 API is handed
        // &request->overlapped as the LPOVERLAPPED for ReadFile, and GetQueuedCompletionStatus
        // later hands that exact same pointer back on completion; since overlapped is the first
        // data member, reinterpret_cast<StreamingRequest*> on that pointer recovers the whole
        // request with no offset arithmetic. This is the standard idiom for embedding per-request
        // context into an IOCP-driven design (used throughout real-world Windows I/O code).
        struct StreamingRequest {
            OVERLAPPED overlapped;
            AsyncFileStreamer::ReadCallback onComplete;
            // Set true when ReadFile failed to even launch this request (see SubmitRead). A
            // synthetic completion packet is posted for it via PostQueuedCompletionStatus rather
            // than invoking onComplete synchronously on the submitting thread -- if the caller's
            // callback re-submits another read (the normal "refill this queue slot" pattern), and
            // that one also fails immediately, invoking onComplete inline here would recurse
            // (SubmitRead -> onComplete -> SubmitRead -> onComplete -> ...) with no bound, which
            // can overflow the stack in milliseconds under a sustained I/O error. Routing every
            // completion, success or failure, through the same worker-thread path guarantees O(1)
            // call depth no matter how many requests fail in a row.
            bool launchFailed = false;
        };

    } // namespace

    AsyncFileStreamer::AsyncFileStreamer(uint32_t workerThreadCount) {
        m_WorkerThreadCount = core::GetDefaultWorkerThreadCount(workerThreadCount);
    }

    AsyncFileStreamer::~AsyncFileStreamer() {
        Close();
    }

    bool AsyncFileStreamer::Open(const std::filesystem::path& filePath) {
        Close(); // Idempotent: no-op if nothing was open.

        m_FileHandle = CreateFileW(
            filePath.wstring().c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);

        if (m_FileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[AsyncFileStreamer] CreateFileW failed for '{}' (GetLastError={})", filePath.string(), GetLastError()));
            return false;
        }

        // Binding the file handle to a fresh completion port with completion key 0 (the third
        // argument) is what makes every ReadFile completion on this handle land in this port,
        // tagged with key 0 -- the value WorkerThreadMain uses to tell a real I/O completion
        // apart from the synthetic shutdown packets Close() posts with kShutdownCompletionKey.
        m_CompletionPort = CreateIoCompletionPort(m_FileHandle, nullptr, 0, m_WorkerThreadCount);
        if (m_CompletionPort == nullptr) {
            LOG_ERROR(std::format(
                "[AsyncFileStreamer] CreateIoCompletionPort failed for '{}' (GetLastError={})", filePath.string(), GetLastError()));
            CloseHandle(m_FileHandle);
            m_FileHandle = INVALID_HANDLE_VALUE;
            return false;
        }

        m_WorkerThreads.reserve(m_WorkerThreadCount);
        for (uint32_t i = 0; i < m_WorkerThreadCount; ++i) {
            m_WorkerThreads.emplace_back([this] { WorkerThreadMain(); });
        }

        return true;
    }

    void AsyncFileStreamer::Close() {
        if (m_CompletionPort != nullptr) {
            // Wake every worker thread exactly once each: GetQueuedCompletionStatus dequeues one
            // packet per call, so N threads need N shutdown packets to all observe one and exit.
            for (size_t i = 0; i < m_WorkerThreads.size(); ++i) {
                PostQueuedCompletionStatus(m_CompletionPort, 0, kShutdownCompletionKey, nullptr);
            }
        }
        for (std::thread& worker : m_WorkerThreads) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_WorkerThreads.clear();

        if (m_CompletionPort != nullptr) {
            CloseHandle(m_CompletionPort);
            m_CompletionPort = nullptr;
        }
        if (m_FileHandle != INVALID_HANDLE_VALUE) {
            CloseHandle(m_FileHandle);
            m_FileHandle = INVALID_HANDLE_VALUE;
        }
    }

    bool AsyncFileStreamer::SubmitRead(uint64_t fileOffset, void* destinationBuffer, uint32_t sizeBytes, ReadCallback onComplete) {
        if (!IsOpen()) {
            LOG_ERROR("[AsyncFileStreamer] SubmitRead called on a closed streamer!");
            return false;
        }
        bool aligned = (fileOffset % kStreamerBlockSizeBytes == 0)
            && (sizeBytes % kStreamerBlockSizeBytes == 0)
            && (reinterpret_cast<uintptr_t>(destinationBuffer) % kStreamerBlockSizeBytes == 0);
        if (!aligned) {
            LOG_ERROR(std::format(
                "[AsyncFileStreamer] SubmitRead: offset={}, size={}, buffer={:#x} must all be multiples of {} bytes (FILE_FLAG_NO_BUFFERING requirement).",
                fileOffset, sizeBytes, reinterpret_cast<uintptr_t>(destinationBuffer), kStreamerBlockSizeBytes));
            return false;
        }

        auto* request = new StreamingRequest{};
        ZeroMemory(&request->overlapped, sizeof(OVERLAPPED));
        request->overlapped.Offset = static_cast<DWORD>(fileOffset & 0xFFFFFFFFu);
        request->overlapped.OffsetHigh = static_cast<DWORD>((fileOffset >> 32) & 0xFFFFFFFFu);
        request->onComplete = std::move(onComplete);

        m_PendingRequests.fetch_add(1, std::memory_order_relaxed);

        BOOL immediateResult = ReadFile(m_FileHandle, destinationBuffer, sizeBytes, nullptr, &request->overlapped);
        DWORD lastError = GetLastError();

        if (immediateResult == FALSE && lastError != ERROR_IO_PENDING) {
            // Genuine launch failure: no real completion packet will ever be queued by the OS for
            // this request, so post a synthetic one ourselves -- see StreamingRequest::
            // launchFailed for why onComplete must never be invoked directly here.
            LOG_ERROR(std::format(
                "[AsyncFileStreamer] ReadFile failed at offset {} (GetLastError={})", fileOffset, lastError));
            request->launchFailed = true;
            if (!PostQueuedCompletionStatus(m_CompletionPort, 0, 0, &request->overlapped)) {
                // Could not even post the synthetic failure packet (e.g. the port itself is
                // gone): this request is unrecoverable through the normal worker-thread path.
                // Finalizing it here is still safe -- PostQueuedCompletionStatus failing is not
                // something a caller's callback can trigger repeatedly by re-submitting, so this
                // path cannot recurse the way a direct onComplete-on-every-failure call could.
                LOG_ERROR("[AsyncFileStreamer] PostQueuedCompletionStatus failed for a synthetic failure packet!");
                ReadCallback failedCallback = std::move(request->onComplete);
                delete request;
                OnRequestFinished();
                if (failedCallback) {
                    failedCallback(false, 0);
                }
            }
            return false;
        }

        // Either ERROR_IO_PENDING (truly asynchronous) or an immediate TRUE (the read finished
        // synchronously despite FILE_FLAG_OVERLAPPED, which does happen, e.g. when the request is
        // fully satisfiable without blocking) -- a completion packet is queued to the IOCP in
        // BOTH cases by default, so WorkerThreadMain handles them identically either way.
        return true;
    }

    void AsyncFileStreamer::WaitForAllPending() {
        std::unique_lock<std::mutex> lock(m_PendingMutex);
        m_PendingCV.wait(lock, [this] { return m_PendingRequests.load(std::memory_order_acquire) == 0; });
    }

    void AsyncFileStreamer::OnRequestFinished() {
        if (m_PendingRequests.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // The counter just reached zero: wake anyone blocked in WaitForAllPending(). The lock
            // is only needed to avoid the classic lost-wakeup race against a waiter that is
            // between checking the predicate and entering condition_variable::wait.
            std::lock_guard<std::mutex> lock(m_PendingMutex);
            m_PendingCV.notify_all();
        }
    }

    void AsyncFileStreamer::WorkerThreadMain() {
        while (true) {
            DWORD bytesTransferred = 0;
            ULONG_PTR completionKey = 0;
            LPOVERLAPPED overlapped = nullptr;

            BOOL ok = GetQueuedCompletionStatus(m_CompletionPort, &bytesTransferred, &completionKey, &overlapped, INFINITE);
            DWORD ioError = ok ? ERROR_SUCCESS : GetLastError();

            // Checked FIRST, before the overlapped == nullptr test below: Close() posts its
            // shutdown packets via PostQueuedCompletionStatus(..., nullptr) -- lpOverlapped is
            // legitimately null for those by construction, so testing completionKey up front is
            // what tells a normal shutdown apart from a genuine port-level failure with the same
            // null-overlapped shape.
            if (completionKey == kShutdownCompletionKey) {
                break; // Close() is shutting this thread down; any in-flight requests were
                       // already drained by WaitForAllPending()-style logic upstream of Close().
            }

            if (overlapped == nullptr) {
                // No packet was actually dequeued (the port handle itself was closed/invalid, or
                // a spurious/aborted wake) -- nothing to process; GetQueuedCompletionStatus
                // returning FALSE with overlapped == nullptr here happens only for port-level
                // failures, which should not occur while Close() still owns a valid handle, so
                // treat it as fatal to this thread rather than looping forever.
                LOG_ERROR(std::format(
                    "[AsyncFileStreamer] GetQueuedCompletionStatus returned no packet (GetLastError={}); worker exiting.", ioError));
                break;
            }

            auto* request = reinterpret_cast<StreamingRequest*>(overlapped);
            // A synthetic packet posted by SubmitRead's immediate-failure path always reports
            // failure, regardless of `ok` (which, for PostQueuedCompletionStatus packets, merely
            // reflects that the dequeue itself succeeded -- it carries no I/O success meaning).
            bool success = (ok != FALSE) && !request->launchFailed;
            AsyncFileStreamer::ReadCallback callback = std::move(request->onComplete);
            delete request;

            // Invoke the callback BEFORE decrementing the pending count. The callback commonly
            // re-submits a new read (the "keep this queue slot full" pattern the benchmark uses),
            // which increments the pending count again from inside SubmitRead -- if the decrement
            // happened first, a concurrent WaitForAllPending() on another thread could observe the
            // count transiently reach zero and return *before* that re-submission is accounted
            // for, letting the caller free/reuse a buffer that a just-launched read is still
            // writing into. Decrementing only after the callback has had its chance to re-submit
            // guarantees the count never spuriously bottoms out mid-stream.
            if (callback) {
                callback(success, bytesTransferred);
            }

            OnRequestFinished();
        }
    }

    void* AsyncFileStreamer::AllocateAlignedBuffer(size_t sizeBytes, size_t alignmentBytes) {
        // _aligned_malloc is the MSVC CRT's aligned allocator; POSIX's posix_memalign/
        // aligned_alloc are not available on this Windows/MSVC target, so there is no portable
        // alternative to reach for here.
        return _aligned_malloc(sizeBytes, alignmentBytes);
    }

    void AsyncFileStreamer::FreeAlignedBuffer(void* buffer) {
        _aligned_free(buffer);
    }

}
