#include "io/StreamingRequestQueue.h"

#include <algorithm>

namespace geometry {

    void StreamingRequestQueue::SubmitFrameRequests(const std::vector<uint32_t>& requestedIDs,
        const std::vector<float>& priorities) {
        for (size_t i = 0; i < requestedIDs.size(); ++i) {
            uint32_t id = requestedIDs[i];
            // insert() returns {iterator, false} if id was already tracked (queued or in flight)
            // -- that repeat report is exactly what must be silently absorbed rather than
            // re-queued.
            auto [it, inserted] = m_TrackedIDs.insert(id);
            if (inserted) {
                m_PendingHeap.push_back(PendingRequest{ id, priorities[i], m_NextSequence++ });
                std::push_heap(m_PendingHeap.begin(), m_PendingHeap.end(), PendingRequestOrder{});
            }
        }
    }

    bool StreamingRequestQueue::PopNextRequest(uint32_t& outID) {
        if (m_PendingHeap.empty()) {
            return false;
        }

        // The id stays in m_TrackedIDs on purpose: it is now "in flight" rather than "queued," but
        // either way a repeat feedback-buffer report for it must still be absorbed until
        // MarkRequestCompleted() explicitly releases it.
        std::pop_heap(m_PendingHeap.begin(), m_PendingHeap.end(), PendingRequestOrder{});
        outID = m_PendingHeap.back().id;
        m_PendingHeap.pop_back();
        return true;
    }

    void StreamingRequestQueue::MarkRequestCompleted(uint32_t id) {
        m_TrackedIDs.erase(id);
    }

    bool StreamingRequestQueue::IsTracked(uint32_t id) const {
        return m_TrackedIDs.contains(id);
    }

}
