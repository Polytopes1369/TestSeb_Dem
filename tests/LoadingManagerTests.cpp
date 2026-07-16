// Standalone, framework-free unit test for core::LoadingManager (src/core/LoadingManager.h).
// Purely CPU-side: zero Vulkan dependency. Exits 0 if every check passes, non-zero otherwise (see
// CMakeLists.txt's CTest registration).
//
// What is validated:
//   1. Fan-out/blocking mode (Submit() with no completion callback + WaitIdle()): many jobs
//      submitted at once actually run concurrently across more than one thread (when the machine
//      has more than one core) and every one of them has genuinely finished by the time WaitIdle()
//      returns -- the mode renderer::GlobalSDFPass::Init() relies on to parallelize its per-entity
//      Mesh SDF bake across cores.
//   2. Frame-budgeted/non-blocking mode (Submit() with a completion callback + PumpCompletions()):
//      completions are only ever invoked from the thread that calls PumpCompletions(), never from
//      a worker thread directly, and PumpCompletions(N) never invokes more than N callbacks in one
//      call even when far more than N jobs have already finished in the background.
//   3. Ordering: for jobs submitted back-to-back that each sleep a random tiny amount, the set of
//      completions drained across multiple PumpCompletions() calls is exactly the set submitted
//      (no job lost, none invoked twice, none invoked before its background work actually ran).
//   4. Shutdown() drains every already-submitted job's background work to completion before the
//      worker threads join (even ones that were still queued, not yet started, at the moment
//      Shutdown() was called) -- no job is ever abandoned mid-flight.
//   5. A completion callback may call Submit() again (chaining) without deadlocking.

#include "core/LoadingManager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

    int g_FailCount = 0;

    void Check(bool condition, const std::string& description) {
        if (!condition) {
            std::cerr << "[FAIL] " << description << std::endl;
            ++g_FailCount;
        }
    }

    void TestFanOutBlockingMode() {
        core::LoadingManager manager;

        constexpr int kJobCount = 64;
        std::atomic<int> completedCount{ 0 };
        std::mutex threadIdsMutex;
        std::set<std::thread::id> observedThreadIds;

        for (int i = 0; i < kJobCount; ++i) {
            manager.Submit([&completedCount, &threadIdsMutex, &observedThreadIds]() {
                {
                    std::lock_guard<std::mutex> lock(threadIdsMutex);
                    observedThreadIds.insert(std::this_thread::get_id());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                completedCount.fetch_add(1, std::memory_order_relaxed);
                });
        }

        manager.WaitIdle();

        Check(completedCount.load() == kJobCount,
            "TestFanOutBlockingMode: WaitIdle() must not return until every submitted job's "
            "background work has finished (expected " + std::to_string(kJobCount) +
            ", got " + std::to_string(completedCount.load()) + ")");
        Check(manager.GetInFlightCount() == 0,
            "TestFanOutBlockingMode: GetInFlightCount() must be 0 once WaitIdle() has returned");

        if (std::thread::hardware_concurrency() > 1) {
            Check(observedThreadIds.size() > 1,
                "TestFanOutBlockingMode: on a multi-core machine, jobs must actually run on more "
                "than one distinct worker thread, not be serialized onto one");
        }
    }

    void TestFrameBudgetedNonBlockingMode() {
        core::LoadingManager manager;
        const std::thread::id mainThreadId = std::this_thread::get_id();

        constexpr int kJobCount = 20;
        std::atomic<int> backgroundRunCount{ 0 };
        std::atomic<bool> anyCallbackRanOnWorkerThread{ false };
        std::vector<int> completionOrder;
        std::mutex completionOrderMutex;

        for (int i = 0; i < kJobCount; ++i) {
            manager.Submit(
                [&backgroundRunCount, i]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1 + (i % 3)));
                    backgroundRunCount.fetch_add(1, std::memory_order_relaxed);
                },
                [&completionOrderMutex, &completionOrder, &anyCallbackRanOnWorkerThread, mainThreadId, i]() {
                    if (std::this_thread::get_id() != mainThreadId) {
                        anyCallbackRanOnWorkerThread.store(true, std::memory_order_relaxed);
                    }
                    std::lock_guard<std::mutex> lock(completionOrderMutex);
                    completionOrder.push_back(i);
                });
        }

        // Let every background job actually finish before pumping, so this test exercises
        // PumpCompletions()'s OWN budget enforcement (not incidentally waiting on slow jobs).
        manager.WaitIdle();
        Check(backgroundRunCount.load() == kJobCount,
            "TestFrameBudgetedNonBlockingMode: every job's background work must have run before "
            "any PumpCompletions() call in this test");
        Check(manager.GetReadyCompletionCount() == static_cast<uint32_t>(kJobCount),
            "TestFrameBudgetedNonBlockingMode: every finished job must have queued its completion, "
            "none invoked early on a worker thread");

        // Drain in small budgeted chunks, exactly like a per-frame caller would.
        constexpr uint32_t kBudgetPerCall = 3;
        uint32_t totalInvoked = 0;
        int callsMade = 0;
        while (totalInvoked < static_cast<uint32_t>(kJobCount)) {
            uint32_t invokedThisCall = manager.PumpCompletions(kBudgetPerCall);
            Check(invokedThisCall <= kBudgetPerCall,
                "TestFrameBudgetedNonBlockingMode: PumpCompletions(N) must never invoke more than N "
                "callbacks in one call");
            totalInvoked += invokedThisCall;
            ++callsMade;
            Check(callsMade < 1000, "TestFrameBudgetedNonBlockingMode: PumpCompletions() loop did not "
                "converge -- likely a stuck/lost completion");
            if (callsMade >= 1000) break;
        }

        Check(totalInvoked == static_cast<uint32_t>(kJobCount),
            "TestFrameBudgetedNonBlockingMode: across all PumpCompletions() calls, exactly every "
            "submitted job's completion must be invoked exactly once");
        Check(!anyCallbackRanOnWorkerThread.load(),
            "TestFrameBudgetedNonBlockingMode: a completion callback must only ever run on the "
            "thread that called PumpCompletions(), never directly on a worker thread");
        Check(manager.GetReadyCompletionCount() == 0,
            "TestFrameBudgetedNonBlockingMode: no completions should remain queued after they have "
            "all been drained");

        std::vector<int> sortedOrder = completionOrder;
        std::sort(sortedOrder.begin(), sortedOrder.end());
        std::vector<int> expected(kJobCount);
        for (int i = 0; i < kJobCount; ++i) expected[i] = i;
        Check(sortedOrder == expected,
            "TestFrameBudgetedNonBlockingMode: every job index 0.." + std::to_string(kJobCount - 1) +
            " must appear in the drained completion set exactly once, none lost or duplicated");
    }

    void TestChainedSubmitFromCompletion() {
        core::LoadingManager manager;
        std::atomic<int> firstJobRan{ 0 };
        std::atomic<int> secondJobRan{ 0 };

        manager.Submit(
            [&firstJobRan]() { firstJobRan.store(1, std::memory_order_relaxed); },
            [&manager, &secondJobRan]() {
                // Chaining: submitting a follow-up job from inside a completion callback must not
                // deadlock (the completed-queue lock is released before the callback runs -- see
                // LoadingManager::PumpCompletions's own comment).
                manager.Submit([&secondJobRan]() { secondJobRan.store(1, std::memory_order_relaxed); });
            });

        manager.WaitIdle(); // Waits for the first job only (the second isn't submitted yet).
        Check(manager.PumpCompletions(1) == 1,
            "TestChainedSubmitFromCompletion: the first job's completion must be ready to pump");
        Check(firstJobRan.load() == 1,
            "TestChainedSubmitFromCompletion: the first job's background work must have run");

        manager.WaitIdle(); // Now waits for the chained second job.
        Check(secondJobRan.load() == 1,
            "TestChainedSubmitFromCompletion: a job submitted from within a completion callback "
            "must still run to completion");
    }

    void TestShutdownDrainsQueuedWork() {
        std::atomic<int> ranCount{ 0 };
        {
            core::LoadingManager manager(2); // Small, fixed pool so most jobs are still queued, not running, when Shutdown() is called.
            constexpr int kJobCount = 50;
            for (int i = 0; i < kJobCount; ++i) {
                manager.Submit([&ranCount]() {
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    ranCount.fetch_add(1, std::memory_order_relaxed);
                    });
            }
            manager.Shutdown(); // Must not return until every one of the 50 jobs above has actually run.
            Check(ranCount.load() == kJobCount,
                "TestShutdownDrainsQueuedWork: Shutdown() must drain every already-submitted job's "
                "background work before returning, even ones still queued (not yet started) at the "
                "moment Shutdown() was called (expected " + std::to_string(kJobCount) + ", got " +
                std::to_string(ranCount.load()) + ")");
        }
        // Destructor's own Shutdown() call must be a safe no-op after an explicit Shutdown() above.
        Check(true, "TestShutdownDrainsQueuedWork: destructor after explicit Shutdown() must not crash");
    }

} // namespace

int main() {
    TestFanOutBlockingMode();
    TestFrameBudgetedNonBlockingMode();
    TestChainedSubmitFromCompletion();
    TestShutdownDrainsQueuedWork();

    if (g_FailCount == 0) {
        std::cout << "[LoadingManagerTests] All checks PASSED.\n";
        return 0;
    }
    std::cerr << "[LoadingManagerTests] " << g_FailCount << " check(s) FAILED.\n";
    return 1;
}
