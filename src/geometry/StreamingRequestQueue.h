#pragma once
// CPU-side streaming request queue fed by renderer::FeedbackBuffer's per-frame readback.
//
// A GPU culling/LOD shader reports a cluster ID once per invocation that finds it missing from
// the page table (see src/shaders/include/feedback_buffer.glsl); since the same missing cluster
// is very likely to be reported again on the very next frame (its LOD requirement and residency
// status typically don't change frame-to-frame while a disk read is still in flight for it), the
// raw per-frame list from FeedbackBuffer::ReadRequestedClusterIDs() is not by itself a usable
// work queue for a disk streamer -- submitting one AsyncFileStreamer read per report would
// resubmit the same read dozens of times before the first one even completes. This class is the
// deduplication layer between the two: SubmitFrameRequests() folds each frame's reports into a
// FIFO of genuinely new requests, tracking what's already queued or in flight so repeat reports
// for the same cluster are silently absorbed until the caller explicitly marks that cluster's
// request as completed (or failed, which is handled identically -- both just make the cluster
// eligible to be requested again).
//
// This class has no knowledge of Vulkan, disk I/O, or the .cache file format -- it only tracks
// cluster IDs (geometry::ClusterIndexEntry::clusterID). The caller is responsible for resolving a
// popped cluster ID to a virtual address (via the cluster index table, see CacheFileManager.h)
// and actually issuing the read (via AsyncFileStreamer / CacheFileManager::ReadClusterDataAsync),
// then binding the loaded page (via renderer::GpuGeometryPagePool::BindPage) before calling
// MarkRequestCompleted().

#include <cstdint>
#include <deque>
#include <unordered_set>
#include <vector>

namespace geometry {

    class StreamingRequestQueue {
    public:
        StreamingRequestQueue() = default;

        // Folds this frame's feedback-buffer readback into the queue: any clusterID not already
        // tracked (queued, or previously submitted and not yet marked completed) is appended to
        // the FIFO and marked tracked. ClusterIDs already tracked are silently ignored -- this is
        // what prevents the same still-missing cluster from being re-queued every single frame
        // while its disk read is in flight.
        void SubmitFrameRequests(const std::vector<uint32_t>& requestedClusterIDs);

        // Pops the oldest untracked-when-submitted request still waiting to be serviced. Returns
        // false (leaving outClusterID unchanged) if the queue is empty. The popped clusterID
        // remains "tracked" (so a repeated feedback-buffer report for it is still absorbed, not
        // re-queued) until the caller calls MarkRequestCompleted() once its disk read (and GPU
        // page bind) has actually finished.
        bool PopNextRequest(uint32_t& outClusterID);

        // Marks `clusterID` as no longer in flight: a future feedback-buffer report for it will
        // be queued again (e.g. because the page it loaded into was since evicted, or because the
        // read failed and must be retried). Safe to call even if `clusterID` is not currently
        // tracked (no-op).
        void MarkRequestCompleted(uint32_t clusterID);

        // True if `clusterID` is currently queued or has been popped but not yet completed.
        bool IsTracked(uint32_t clusterID) const;

        size_t PendingCount() const { return m_PendingQueue.size(); }
        size_t TrackedCount() const { return m_TrackedClusterIDs.size(); }

    private:
        std::deque<uint32_t> m_PendingQueue;
        std::unordered_set<uint32_t> m_TrackedClusterIDs;
    };

}
