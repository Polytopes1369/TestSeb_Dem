#pragma once
// Generic, reusable multithreaded loading/background-work manager: a small fixed pool of worker
// threads (hardware_concurrency-based, matching io::AsyncFileStreamer's own convention) that runs
// arbitrary CPU-bound jobs off the main thread, plus a frame-budgeted completion drain so a caller
// can integrate a finished job's result on the main thread without ever blocking a frame -- the
// same "worker thread does the heavy lifting, main thread only pays for a small budgeted drain per
// frame" discipline this codebase's own geometry disk-streaming stack (io::AsyncFileStreamer /
// renderer::GeometryStreamingCoordinator) already uses for cluster paging, generalized here to ANY
// CPU-bound background job (procedural bakes, mesh/SDF generation, texture synthesis, future
// terrain/tree/foliage generators -- see CLAUDE.md's stated procedural-generation scope), not just
// disk I/O. This plays the same architectural role Unreal Engine 5.8's Task Graph System plays as
// the shared low-level primitive underneath domain-specific systems like the Async Loading Thread:
// this class never touches Vulkan or any engine-specific resource itself -- it only decides WHEN a
// finished background job's result is handed back to the calling thread and HOW MANY per call; the
// caller's own completion callback (or, for the blocking path, the caller's own post-WaitIdle()
// code) does whatever GPU/engine-specific integration it needs from there.
//
// --- Two ways to consume a submitted job ---
// 1. Frame-budgeted / non-blocking (the "spread across multiple frames" mode): Submit() with a
//    non-null `onMainThreadComplete`, then call PumpCompletions(budget) once per frame (from the
//    single Vulkan-recording thread) to invoke up to `budget` queued completions. A completion
//    invoked this way may safely record Vulkan commands into a caller-supplied, already-open
//    command buffer if the caller's own Pump*() wrapper threads one through (see
//    renderer::GlobalSDFPass::PumpAsyncBakes for the pattern) -- PumpCompletions() itself stays
//    Vulkan-agnostic; it never touches a command buffer, only invokes plain callbacks.
// 2. Fan-out / blocking (the "use every core, but I still need the result before I continue" mode,
//    e.g. an Init()-time bake all of a class's callers assume is complete before they proceed):
//    Submit() with `onMainThreadComplete == nullptr` (each job's background work writes its result
//    directly into its own pre-sized, disjoint output slot -- safe without extra locking, exactly
//    like a parallel-for), then WaitIdle() to block until every submitted job's background work
//    has actually finished running on some worker thread.
// Both modes share the same worker pool and the same Submit() call -- a caller decides per call
// which discipline it needs.
//
// --- Threading contract ---
// Submit() may be called from any thread, including from inside a completion callback (to chain a
// follow-up job). `backgroundWork` always runs on a worker thread, never the calling thread.
// `onMainThreadComplete` is never invoked directly from a worker thread -- it is only ever invoked
// later, from whichever thread calls PumpCompletions(), so it is exactly as safe to touch
// non-thread-safe engine state from a completion callback as it would be from that calling thread
// directly (matching io::AsyncFileStreamer's own ReadCallback contract, generalized).

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace core {

    class LoadingManager {
    public:
        // `workerThreadCount == 0` auto-selects std::thread::hardware_concurrency() (falling back
        // to 1 if the platform can't report it), matching io::AsyncFileStreamer's own convention so
        // this pool never oversubscribes the machine's actual core count.
        explicit LoadingManager(uint32_t workerThreadCount = 0);
        ~LoadingManager();

        LoadingManager(const LoadingManager&) = delete;
        LoadingManager& operator=(const LoadingManager&) = delete;

        // Enqueues `backgroundWork` to run on a worker thread as soon as one is free. If
        // `onMainThreadComplete` is non-null, it is queued (not invoked) once `backgroundWork`
        // returns, ready for a future PumpCompletions() call -- see the class comment's two
        // consumption modes. Safe to call from any thread, including from within a running
        // `backgroundWork` or a completion callback.
        void Submit(std::function<void()> backgroundWork, std::function<void()> onMainThreadComplete = nullptr);

        // Invokes up to `maxCompletionsThisCall` queued `onMainThreadComplete` callbacks, in the
        // order their background work finished, on the CALLING thread. Returns the number actually
        // invoked (may be less than requested if fewer are ready, including zero). A completion
        // invoked here may itself call Submit() again (e.g. to chain follow-up work) without
        // deadlocking -- the completed-queue lock is never held while a callback runs.
        uint32_t PumpCompletions(uint32_t maxCompletionsThisCall);

        // Blocks the calling thread until every job submitted so far (with or without a completion
        // callback) has finished running its `backgroundWork` on some worker thread. Does NOT wait
        // for queued `onMainThreadComplete` callbacks to be pumped -- those are a separate,
        // caller-driven step (PumpCompletions()); a job submitted with `onMainThreadComplete ==
        // nullptr` has nothing left to wait for once its background work returns, which is exactly
        // the fan-out/blocking mode's contract (see class comment).
        void WaitIdle();

        // Jobs submitted but not yet finished running (queued or currently executing on a worker).
        uint32_t GetInFlightCount() const { return m_InFlight.load(std::memory_order_acquire); }
        // Completed jobs whose `onMainThreadComplete` is queued, waiting for a PumpCompletions() call.
        uint32_t GetReadyCompletionCount() const;

        // Signals every worker thread to stop once the job queue is fully drained (every job
        // submitted before this call, including ones still queued, WILL still run to completion --
        // see the .cpp's worker loop -- matching io::AsyncFileStreamer::Close()'s own "no abandoned
        // in-flight work" guarantee), then joins them. Any `onMainThreadComplete` callbacks queued
        // by jobs that finish during shutdown are simply discarded when this instance is destroyed
        // -- safe, since a callback is only ever a plain std::function, never itself holding a
        // Vulkan handle that needs explicit destruction (the caller's own Pump*() wrapper owns
        // that). Safe to call multiple times (idempotent) and safe to skip -- the destructor calls
        // this too.
        void Shutdown();

    private:
        void WorkerThreadMain();

        struct Job {
            std::function<void()> backgroundWork;
            std::function<void()> onMainThreadComplete;
        };

        std::vector<std::thread> m_WorkerThreads;
        std::atomic<bool> m_ShuttingDown{ false };

        std::mutex m_QueueMutex;
        std::condition_variable m_QueueCV;
        std::deque<Job> m_JobQueue; // Guarded by m_QueueMutex.

        mutable std::mutex m_CompletedMutex; // mutable: GetReadyCompletionCount() is const but must lock it.
        std::deque<std::function<void()>> m_CompletedCallbacks; // Guarded by m_CompletedMutex.

        // Jobs submitted but not yet finished running their backgroundWork -- incremented in
        // Submit(), decremented once a worker's call to backgroundWork() returns. WaitIdle() blocks
        // on this reaching zero, independent of m_CompletedCallbacks (see WaitIdle()'s own comment).
        std::atomic<uint32_t> m_InFlight{ 0 };
        std::mutex m_InFlightMutex;
        std::condition_variable m_InFlightCV;
    };

}
