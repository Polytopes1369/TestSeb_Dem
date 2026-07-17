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
//
// Priority (UE 5.8 Nanite/VT streaming parity): requests are serviced highest-priority-first, not
// strict FIFO -- a higher `priority` value in SubmitFrameRequests() is popped before a lower one,
// so a near/coarse/high-impact request queued this frame does not wait behind a far/fine-detail
// one queued earlier. Ties (equal priority) fall back to submission order, so requests of equal
// importance are never reordered arbitrarily and none can starve indefinitely under sustained
// same-priority pressure. Each of the four callers (geometry pages, virtual-texture tiles, virtual
// shadow map pages, terrain/decal VT pages) derives `priority` from whatever signal it already has
// cheaply available -- DAG level, mip level, clipmap level -- see their own call sites.

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace geometry {

    class StreamingRequestQueue {
    public:
        StreamingRequestQueue() = default;

        // Folds this frame's feedback-buffer readback into the queue: any id not already tracked
        // (queued, or previously submitted and not yet marked completed) is inserted with its
        // matching `priorities[i]`. IDs already tracked are silently ignored -- this is what
        // prevents the same still-missing request from being re-queued every single frame while
        // its disk read is in flight. `priorities` must be the same length as `requestedIDs`,
        // index-aligned; a higher value is serviced sooner (see class comment).
        void SubmitFrameRequests(const std::vector<uint32_t>& requestedIDs, const std::vector<float>& priorities);

        // Pops the highest-priority request still waiting to be serviced (ties broken by
        // submission order -- see class comment). Returns false (leaving outID unchanged) if the
        // queue is empty. The popped id remains "tracked" (so a repeated feedback-buffer report
        // for it is still absorbed, not re-queued) until the caller calls MarkRequestCompleted()
        // once its disk read (and GPU page bind) has actually finished.
        bool PopNextRequest(uint32_t& outID);

        // Marks `id` as no longer in flight: a future feedback-buffer report for it will be
        // queued again (e.g. because the page it loaded into was since evicted, or because the
        // read failed and must be retried). Safe to call even if `id` is not currently tracked
        // (no-op).
        void MarkRequestCompleted(uint32_t id);

        // True if `id` is currently queued or has been popped but not yet completed.
        bool IsTracked(uint32_t id) const;

        size_t PendingCount() const { return m_PendingHeap.size(); }
        size_t TrackedCount() const { return m_TrackedIDs.size(); }

    private:
        struct PendingRequest {
            uint32_t id;
            float priority;
            uint64_t sequence; // Submission order, for FIFO tie-break among equal priority.
        };

        // std::push_heap/pop_heap order: "a < b" must mean "b is serviced before a". Higher
        // priority first; among equal priority, lower sequence (earlier submitted) first.
        struct PendingRequestOrder {
            bool operator()(const PendingRequest& a, const PendingRequest& b) const {
                if (a.priority != b.priority) {
                    return a.priority < b.priority;
                }
                return a.sequence > b.sequence;
            }
        };

        std::vector<PendingRequest> m_PendingHeap;
        std::unordered_set<uint32_t> m_TrackedIDs;
        uint64_t m_NextSequence = 0;
    };

}
