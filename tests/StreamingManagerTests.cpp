// Standalone, framework-free unit test for world::StreamingManager (src/world/StreamingManager.h):
// drives a real core::LoadingManager worker pool through a synthetic IWorldCellLoader that just
// records which cells were (un)loaded, and verifies the detail/HLOD/unload radii and the resulting
// cell state machine converge correctly. Exits 0 if every check passes, non-zero otherwise --
// registered with CTest (see the top-level CMakeLists.txt), matching this project's existing
// tests/*.cpp convention.
//
// Draining is done deterministically (Update() + workerPool.WaitIdle(), repeated until the pending
// queue and in-flight count both reach zero) rather than sleep-based polling, so this test is not
// flaky under CI/machine load.

#include "world/StreamingManager.h"
#include "world/StreamingTypes.h"
#include "core/LoadingManager.h"

#include <iostream>
#include <mutex>
#include <set>
#include <string>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    // Records every load/unload call it receives, guarded by a mutex since StreamingManager may
    // call these concurrently from multiple worker threads for different cells.
    class RecordingLoader final : public world::IWorldCellLoader {
    public:
        void LoadCellFullDetail(const world::CellCoord& coord) override {
            std::lock_guard<std::mutex> lock(mutex_);
            fullDetailLoads_.insert(coord);
            hlodLoads_.erase(coord);
        }
        void LoadCellHlod(const world::CellCoord& coord) override {
            std::lock_guard<std::mutex> lock(mutex_);
            hlodLoads_.insert(coord);
            fullDetailLoads_.erase(coord);
        }
        void UnloadCell(const world::CellCoord& coord) override {
            std::lock_guard<std::mutex> lock(mutex_);
            fullDetailLoads_.erase(coord);
            hlodLoads_.erase(coord);
            unloadCount_++;
        }

        bool IsFullDetailLoaded(const world::CellCoord& coord) {
            std::lock_guard<std::mutex> lock(mutex_);
            return fullDetailLoads_.count(coord) > 0;
        }
        bool IsHlodLoaded(const world::CellCoord& coord) {
            std::lock_guard<std::mutex> lock(mutex_);
            return hlodLoads_.count(coord) > 0;
        }
        size_t LoadedCount() {
            std::lock_guard<std::mutex> lock(mutex_);
            return fullDetailLoads_.size() + hlodLoads_.size();
        }
        int UnloadCount() {
            std::lock_guard<std::mutex> lock(mutex_);
            return unloadCount_;
        }

    private:
        std::mutex mutex_;
        std::set<world::CellCoord, bool(*)(const world::CellCoord&, const world::CellCoord&)> fullDetailLoads_{
            [](const world::CellCoord& a, const world::CellCoord& b) { return a.x != b.x ? a.x < b.x : a.z < b.z; } };
        std::set<world::CellCoord, bool(*)(const world::CellCoord&, const world::CellCoord&)> hlodLoads_{
            [](const world::CellCoord& a, const world::CellCoord& b) { return a.x != b.x ? a.x < b.x : a.z < b.z; } };
        int unloadCount_ = 0;
    };

    // Dispatches queued requests and waits for them to actually finish, repeatedly, until nothing
    // is pending or in flight (or maxIterations is exceeded, in which case the test that called
    // this will simply see stale state and fail its own checks -- there is no separate timeout
    // failure path needed).
    void DrainStreaming(world::StreamingManager& mgr, core::LoadingManager& pool, int maxIterations = 1000) {
        for (int i = 0; i < maxIterations; ++i) {
            mgr.Update();
            pool.WaitIdle();
            if (mgr.GetPendingQueueLength() == 0 && mgr.GetInFlightCount() == 0) return;
        }
    }

    void TestWorldToCellAndCellCenter() {
        RecordingLoader loader;
        core::LoadingManager pool(1);
        world::StreamingManager mgr(10.0f, loader, pool, 4);

        world::CellCoord coord = mgr.WorldToCell({ 23.5f, 0.0f, -7.0f });
        Check(coord.x == 2, "WorldToCell: expected x=2 for world x=23.5 at cellSize 10");
        Check(coord.z == -1, "WorldToCell: expected z=-1 for world z=-7 at cellSize 10");

        maths::vec3 center = mgr.CellCenter(coord);
        Check(center.x > 23.5f && center.x < 30.0f, "CellCenter: center should be inside the cell that contains the original world x");
    }

    void TestDetailAndHlodRadii() {
        RecordingLoader loader;
        core::LoadingManager pool(2);
        world::StreamingManager mgr(50.0f, loader, pool, 2);

        world::StreamingSource source;
        source.position = { 25.0f, 0.0f, 25.0f };
        source.detailLoadRadius = 60.0f;
        source.hlodLoadRadius = 200.0f;

        mgr.UpdateStreamingSources({ source });
        DrainStreaming(mgr, pool);

        world::CellCoord nearCell = mgr.WorldToCell(source.position);
        Check(mgr.GetCellState(nearCell) == world::CellStreamingState::LoadedActive, "cell at the source's own position should end up LoadedActive");
        Check(mgr.GetCellRepresentation(nearCell) == world::CellRepresentation::FullDetail, "cell at the source's own position should load at FullDetail (within detailLoadRadius)");
        Check(loader.IsFullDetailLoaded(nearCell), "RecordingLoader should have received a LoadCellFullDetail call for the near cell");

        // A cell well outside detailLoadRadius (60) but inside hlodLoadRadius (200): e.g. 150
        // world units east of the source, along +X.
        world::CellCoord hlodCell = mgr.WorldToCell({ source.position.x + 150.0f, 0.0f, source.position.z });
        Check(mgr.GetCellState(hlodCell) == world::CellStreamingState::LoadedActive, "cell within hlodLoadRadius but outside detailLoadRadius should end up LoadedActive");
        Check(mgr.GetCellRepresentation(hlodCell) == world::CellRepresentation::HLOD, "cell within hlodLoadRadius but outside detailLoadRadius should load as HLOD");
        Check(loader.IsHlodLoaded(hlodCell), "RecordingLoader should have received a LoadCellHlod call for the ring cell");

        // A cell far outside hlodLoadRadius must never have been touched at all.
        world::CellCoord farCell = mgr.WorldToCell({ source.position.x + 5000.0f, 0.0f, source.position.z });
        Check(mgr.GetCellState(farCell) == world::CellStreamingState::Unloaded, "a cell far outside every radius must remain Unloaded (never tracked)");
        Check(!loader.IsFullDetailLoaded(farCell) && !loader.IsHlodLoaded(farCell), "a cell far outside every radius must never be loaded");
    }

    void TestUnloadWhenSourceMovesAway() {
        RecordingLoader loader;
        core::LoadingManager pool(2);
        world::StreamingManager mgr(50.0f, loader, pool, 2);

        world::StreamingSource source;
        source.position = { 0.0f, 0.0f, 0.0f };
        source.detailLoadRadius = 40.0f;
        source.hlodLoadRadius = 40.0f; // Equal radii: keeps this test to a single representation, isolating the unload path from the HLOD<->FullDetail swap path (covered by TestDetailAndHlodRadii).

        mgr.UpdateStreamingSources({ source });
        DrainStreaming(mgr, pool);

        world::CellCoord loadedCell = mgr.WorldToCell(source.position);
        Check(mgr.GetCellState(loadedCell) == world::CellStreamingState::LoadedActive, "setup: cell at the source should be loaded before it moves away");
        size_t trackedBeforeMove = mgr.GetTrackedCellCount();

        // Move the only source far away: every previously-loaded cell should become desired=None
        // and get unloaded.
        world::StreamingSource movedSource = source;
        movedSource.position = { 100000.0f, 0.0f, 100000.0f };
        mgr.UpdateStreamingSources({ movedSource });
        DrainStreaming(mgr, pool);

        Check(mgr.GetCellState(loadedCell) == world::CellStreamingState::Unloaded, "cell should be Unloaded again once its only source moves far away");
        Check(mgr.GetCellRepresentation(loadedCell) == world::CellRepresentation::None, "cell's representation should be None again once unloaded");
        Check(loader.UnloadCount() > 0, "RecordingLoader should have received at least one UnloadCell call");
        Check(mgr.GetTrackedCellCount() >= trackedBeforeMove, "cell records are never erased -- tracked count must not shrink even after an unload");
    }

}

int main() {
    TestWorldToCellAndCellCenter();
    TestDetailAndHlodRadii();
    TestUnloadWhenSourceMovesAway();

    if (g_failCount == 0) {
        std::cout << "[PASS] All StreamingManager checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
