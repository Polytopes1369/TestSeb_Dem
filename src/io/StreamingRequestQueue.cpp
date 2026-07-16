#include "io/StreamingRequestQueue.h"

namespace geometry {

    void StreamingRequestQueue::SubmitFrameRequests(const std::vector<uint32_t>& requestedClusterIDs) {
        for (uint32_t clusterID : requestedClusterIDs) {
            // insert() returns {iterator, false} if clusterID was already tracked (queued or
            // in flight) -- that repeat report is exactly what must be silently absorbed rather
            // than re-queued.
            auto [it, inserted] = m_TrackedClusterIDs.insert(clusterID);
            if (inserted) {
                m_PendingQueue.push_back(clusterID);
            }
        }
    }

    bool StreamingRequestQueue::PopNextRequest(uint32_t& outClusterID) {
        if (m_PendingQueue.empty()) {
            return false;
        }

        // The clusterID stays in m_TrackedClusterIDs on purpose: it is now "in flight" rather
        // than "queued," but either way a repeat feedback-buffer report for it must still be
        // absorbed until MarkRequestCompleted() explicitly releases it.
        outClusterID = m_PendingQueue.front();
        m_PendingQueue.pop_front();
        return true;
    }

    void StreamingRequestQueue::MarkRequestCompleted(uint32_t clusterID) {
        m_TrackedClusterIDs.erase(clusterID);
    }

    bool StreamingRequestQueue::IsTracked(uint32_t clusterID) const {
        return m_TrackedClusterIDs.contains(clusterID);
    }

}
