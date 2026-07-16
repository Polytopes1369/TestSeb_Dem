#include "core/LoadingManager.h"
#include "core/ThreadingUtil.h"

namespace core {

    LoadingManager::LoadingManager(uint32_t workerThreadCount) {
        uint32_t count = GetDefaultWorkerThreadCount(workerThreadCount);

        m_WorkerThreads.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            m_WorkerThreads.emplace_back(&LoadingManager::WorkerThreadMain, this);
        }
    }

    LoadingManager::~LoadingManager() {
        Shutdown();
    }

    void LoadingManager::Submit(std::function<void()> backgroundWork, std::function<void()> onMainThreadComplete) {
        m_InFlight.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(m_QueueMutex);
            m_JobQueue.push_back(Job{ std::move(backgroundWork), std::move(onMainThreadComplete) });
        }
        m_QueueCV.notify_one();
    }

    void LoadingManager::WorkerThreadMain() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(m_QueueMutex);
                m_QueueCV.wait(lock, [this] { return !m_JobQueue.empty() || m_ShuttingDown.load(std::memory_order_acquire); });
                if (m_JobQueue.empty()) {
                    // Only reachable when m_ShuttingDown is set (the predicate above guarantees
                    // one of the two is true) -- every job submitted before Shutdown() is called
                    // has already been popped and run by this point, so it is safe to exit now.
                    return;
                }
                job = std::move(m_JobQueue.front());
                m_JobQueue.pop_front();
            }

            if (job.backgroundWork) {
                job.backgroundWork();
            }

            if (job.onMainThreadComplete) {
                std::lock_guard<std::mutex> lock(m_CompletedMutex);
                m_CompletedCallbacks.push_back(std::move(job.onMainThreadComplete));
            }

            {
                // Notify under the lock so a concurrent WaitIdle() call cannot miss this wakeup
                // between checking the predicate and starting its own wait (the classic
                // lost-wakeup hazard condition_variable::wait's predicate overload guards against
                // internally, but only for waiters already inside wait() -- the fetch_sub/notify
                // pair must still be atomic with respect to a waiter's initial predicate check).
                std::lock_guard<std::mutex> lock(m_InFlightMutex);
                m_InFlight.fetch_sub(1, std::memory_order_acq_rel);
            }
            m_InFlightCV.notify_all();
        }
    }

    uint32_t LoadingManager::PumpCompletions(uint32_t maxCompletionsThisCall) {
        uint32_t invoked = 0;
        while (invoked < maxCompletionsThisCall) {
            std::function<void()> callback;
            {
                std::lock_guard<std::mutex> lock(m_CompletedMutex);
                if (m_CompletedCallbacks.empty()) {
                    break;
                }
                callback = std::move(m_CompletedCallbacks.front());
                m_CompletedCallbacks.pop_front();
            }
            // Invoked with the completed-queue lock released, so the callback may itself call
            // Submit() (or even PumpCompletions() again) without deadlocking.
            callback();
            ++invoked;
        }
        return invoked;
    }

    void LoadingManager::WaitIdle() {
        std::unique_lock<std::mutex> lock(m_InFlightMutex);
        m_InFlightCV.wait(lock, [this] { return m_InFlight.load(std::memory_order_acquire) == 0; });
    }

    uint32_t LoadingManager::GetReadyCompletionCount() const {
        std::lock_guard<std::mutex> lock(m_CompletedMutex);
        return static_cast<uint32_t>(m_CompletedCallbacks.size());
    }

    void LoadingManager::Shutdown() {
        m_ShuttingDown.store(true, std::memory_order_release);
        m_QueueCV.notify_all();
        for (std::thread& worker : m_WorkerThreads) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        m_WorkerThreads.clear();

        // Reset so a destroyed-then-reconstructed-in-place instance (not currently done anywhere,
        // but cheap to keep correct) or a defensive double-Shutdown() call behaves sanely.
        m_ShuttingDown.store(false, std::memory_order_relaxed);
        m_JobQueue.clear();
        m_CompletedCallbacks.clear();
        m_InFlight.store(0, std::memory_order_relaxed);
    }

}
