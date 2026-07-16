#pragma once
// Wires the previously self-contained, never-driven async streaming stack (geometry::
// AsyncFileStreamer, geometry::StreamingRequestQueue, renderer::FeedbackBuffer -- see
// ClusterRenderPipeline's old class comment, which called this "future work") into the live
// per-frame path: a residency miss reported by ClusterLODResidencyFallback.comp (via
// renderer::ClusterLODSelectionPass::GetFeedbackBuffer()) is read back, deduplicated, turned into
// a real 4 KB-aligned disk read, and -- once that read completes -- bound into
// renderer::GpuGeometryPagePool and decompressed, all without ever blocking a frame
// (vkQueueWaitIdle) or missing this engine's one-submit-per-frame guarantee.
//
// --- Threading ---
// geometry::AsyncFileStreamer's completion callback fires on one of ITS OWN worker threads, never
// the main thread that submitted the read (see AsyncFileStreamer.h) -- and Vulkan command
// recording in this codebase only ever happens on the main thread. The callback registered here
// therefore does nothing but push a small POD into a mutex-guarded queue; every Vulkan-facing
// action (BindPageEvictingIfFull, DecompressPage) happens later, back on the main thread, inside
// ProcessFeedbackAndDrainCompletions().
//
// --- Per-frame sequence a caller must record, in order (both on the main thread) ---
//   1. ProcessFeedbackAndDrainCompletions(cmd) -- CPU-side triage (reads back LAST frame's misses,
//      issues up to kMaxNewReadsPerFrame new disk reads, drains up to kMaxPagesBoundPerFrame
//      finished ones) plus the actual BindPageEvictingIfFull/DecompressPage command recording for
//      whatever drained this frame. Call this EARLY in the frame (before
//      renderer::ClusterLODSelectionPass's own dispatch), so any page bound this frame is already
//      resident by the time this frame's residency checks run -- matching this class's contract
//      that "reads back LAST frame's misses" implies a one-frame lag between a miss and its data
//      landing, never worse.
//   2. Elsewhere in the frame (after ClusterLODResidencyFallback.comp -- the shader that actually
//      calls RequestClusterResidency() -- has run): the caller still owns calling
//      renderer::FeedbackBuffer::RecordReadback() itself (see ClusterLODSelectionPass::
//      GetFeedbackBuffer()) so THIS frame's misses are captured for next frame's step 1 -- this
//      class does not record that barrier/copy itself, only consumes ReadRequestedClusterIDs()
//      from it.
//
// Exactly like every other piece of this Nanite-style pipeline, this class is a self-contained
// building block -- Init()/Shutdown()/per-frame ProcessFeedbackAndDrainCompletions() only.

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "io/AsyncFileStreamer.h"
#include "io/StreamingRequestQueue.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class FeedbackBuffer;
    class GpuGeometryPagePool;
    class GeometryDecompressionPass;

    class GeometryStreamingCoordinator {
    public:
        GeometryStreamingCoordinator() = default;

        GeometryStreamingCoordinator(const GeometryStreamingCoordinator&) = delete;
        GeometryStreamingCoordinator& operator=(const GeometryStreamingCoordinator&) = delete;

        // At most this many new disk reads are issued per frame (bounds this frame's I/O
        // submission cost, not the total number of misses -- excess misses simply wait in
        // geometry::StreamingRequestQueue for a future frame's budget).
        // Tuned for the original ~1-2K-cluster scene; config::FLOOR_VERTEX_SPACING = 0.5f raised
        // the floor alone to ~5-9K clusters, and at the old budget of 4-8/frame full residency
        // took thousands of frames (tens of seconds) to converge, reading as "holes that never
        // load". Each unit of budget costs one kPageSizeBytes (4 KB) raw buffer and one staging-
        // ring slot -- trivial (this jump costs ~224 KB of host memory total) -- so there is no
        // capacity reason to keep these low; only I/O submission / command-recording cost per
        // frame, which stays cheap even at this size for a local SSD-backed cache file.
        static constexpr uint32_t kMaxNewReadsPerFrame = 64;
        // At most this many completed reads are bound/decompressed per frame (bounds this frame's
        // extra command-recording cost; a completed read that misses this budget just waits in
        // m_CompletedReads for the next frame -- it is never dropped).
        static constexpr uint32_t kMaxPagesBoundPerFrame = 64;
        // Number of concurrent in-flight disk reads this coordinator allows -- also the size of
        // the raw aligned I/O destination buffer pool.
        static constexpr uint32_t kMaxInFlightReads = 64;

        // Opens `cacheFilePath` for async reads (geometry::AsyncFileStreamer) and copies
        // `indexEntries` (needed every frame to resolve a clusterID -> virtualAddress/bounds,
        // since geometry::ClusterIndexEntry::clusterID is a dense 0-based index matching this
        // vector's own position -- see geometry::VirtualGeometryCacheTest.cpp's documented
        // invariant). Allocates the raw aligned I/O buffer pool and the small VMA-mapped staging
        // ring BindPage's staging-buffer parameter needs.
        void Init(VkDevice device, VmaAllocator allocator,
            const std::filesystem::path& cacheFilePath,
            const std::vector<geometry::ClusterIndexEntry>& indexEntries);

        void Shutdown();

        // See the class comment's per-frame step 1. `feedbackBuffer`/`pagePool`/`decompressionPass`
        // are borrowed (owned by renderer::ClusterLODSelectionPass / renderer::ClusterRenderPipeline
        // respectively) -- not stored beyond this call.
        void ProcessFeedbackAndDrainCompletions(VkCommandBuffer cmd, FeedbackBuffer& feedbackBuffer,
            GpuGeometryPagePool& pagePool, GeometryDecompressionPass& decompressionPass);

        uint32_t GetPendingRequestCount() const { return static_cast<uint32_t>(m_RequestQueue.PendingCount()); }
        uint32_t GetInFlightReadCount() const { return m_Streamer.PendingRequestCount(); }
        uint64_t GetTotalBytesCompleted() const { return m_TotalBytesCompleted.load(std::memory_order_relaxed); }

    private:
        struct CompletedRead {
            uint32_t clusterID;
            uint32_t slotIndex;
            uint32_t bytesTransferred;
            bool success;
        };

        static constexpr uint32_t kInvalidSlot = 0xFFFFFFFFu;
        uint32_t FindFreeSlot() const;

        VkDevice m_Device = VK_NULL_HANDLE;

        geometry::AsyncFileStreamer m_Streamer;
        geometry::StreamingRequestQueue m_RequestQueue;

        // Owned copy: geometry::ClusterIndexEntry::clusterID is a dense 0-based index matching
        // this vector's own array position, so a popped clusterID resolves to its entry via a
        // plain index -- no separate map needed. Must outlive every reference the Init() caller's
        // own copy (a local variable in renderer::ClusterRenderPipeline::Init) does not.
        std::vector<geometry::ClusterIndexEntry> m_IndexEntries;

        // I/O destination slots: raw, sector-aligned host buffers (FILE_FLAG_NO_BUFFERING
        // requires this -- see AsyncFileStreamer.h), reused round-robin. Slot busy/free state is
        // only ever touched by the main thread (issuing a read marks it busy; draining a
        // completion frees it) -- the worker-thread completion callback never touches
        // m_SlotBusy, only the mutex-guarded m_CompletedReads queue below.
        std::array<void*, kMaxInFlightReads> m_RawBuffers{};
        std::array<bool, kMaxInFlightReads> m_SlotBusy{};

        // Small staging ring (VMA CPU_ONLY mapped, kMaxPagesBoundPerFrame pages) -- the only
        // bridge between a completed disk read's raw buffer and
        // renderer::GpuGeometryPagePool::BindPage's VkBuffer-typed staging parameter.
        GpuBuffer m_StagingRing;

        std::mutex m_CompletedMutex;
        std::vector<CompletedRead> m_CompletedReads; // Pushed by worker threads, drained by the main thread.
        std::atomic<uint64_t> m_TotalBytesCompleted{ 0 };
    };

}
